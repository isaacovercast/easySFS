"""
Contains Spectrum object, which represents frequency spectra. I poached this
code from dadi (Gutenkunst et al 2008) and retained only what's necessary
for easySFS to function. Any errors introduced in SFS calculation or construction
are 100% my fault. - 1/28/23 IAO

If you use easySFS in your research please cite:
RN Gutenkunst, RD Hernandez, SH Williamson, CD Bustamante "Inferring the joint
demographic history of multiple populations from multidimensional SNP data"
PLoS Genetics 5:e1000695 (2009).

Copyright (c) 2008, Ryan Gutenkunst

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

a. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
b. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
c. Neither the name of the Cornell University nor the names of the
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import collections, gzip, operator, os, sys
import numpy as np
from numpy import newaxis as nuax
from scipy.special import gammaln


## From dadi.Numerics
def reverse_array(arr):
    """
    Reverse an array along all axes, so arr[i,j] -> arr[-(i+1),-(j+1)].
    """
    reverse_slice = tuple(slice(None, None, -1) for ii in arr.shape)
    return arr[reverse_slice]

_projection_cache = {}
def _lncomb(N,k):
    """
    Log of N choose k.
    """
    return gammaln(N+1) - gammaln(k+1) - gammaln(N-k+1)

def _cached_projection(proj_to, proj_from, hits):
    """
    Coefficients for projection from a different fs size.

    proj_to: Numper of samples to project down to.
    proj_from: Numper of samples to project from.
    hits: Number of derived alleles projecting from.
    """
    key = (proj_to, proj_from, hits)
    try:
        return _projection_cache[key]
    except KeyError:
        pass

    if np.isscalar(proj_to) and np.isscalar(proj_from)\
       and proj_from < proj_to:
        # Short-circuit calculation.
        contrib = np.zeros(proj_to+1)
    else:
        # We set numpy's error reporting so that it will ignore underflows, 
        # because those just imply that contrib is 0.
        previous_err_state = np.seterr(under='ignore', divide='raise',
                                          over='raise', invalid='raise')
        proj_hits = np.arange(proj_to+1)
        # For large sample sizes, we need to do the calculation in logs, and it
        # is accurate enough for small sizes as well.
        lncontrib = _lncomb(proj_to,proj_hits)
        lncontrib += _lncomb(proj_from-proj_to,hits-proj_hits)
        lncontrib -= _lncomb(proj_from, hits)
        contrib = np.exp(lncontrib)
        np.seterr(**previous_err_state)
    _projection_cache[key] = contrib
    return contrib


class Spectrum(np.ma.masked_array):
    """
    Represents a frequency spectrum.

    Spectra are represented by masked arrays. The masking allows us to ignore
    specific entries in the spectrum. Most often, these are the absent and fixed
    categories.

    The constructor has the format:
        fs = dadi.Spectrum(data, mask, mask_corners, data_folded, check_folding,
                           pop_ids, extrap_x)
        
        data: The frequency spectrum data
        mask: An optional array of the same size as data. 'True' entires in
              this array are masked in the Spectrum. These represent missing
              data categories. (For example, you may not trust your singleton
              SNP calling.)
        mask_corners: If True (default), the 'observed in none' and 'observed 
                      in all' entries of the FS will be masked. Typically these
                      entries are unobservable, and dadi cannot reliably
                      calculate them, so you will almost always want
                      mask_corners=True.g
        data_folded: If True, it is assumed that the input data is folded. An
                     error will be raised if the input data and mask are not
                     consistent with a folded Spectrum.
        check_folding: If True and data_folded=True, the data and mask will be
                       checked to ensure they are consistent with a folded
                       Spectrum. If they are not, a warning will be printed.
        pop_ids: Optional list of strings containing the population labels.
        extrap_x: Optional floating point value specifying x value to use
                  for extrapolation.
    """
    def __new__(subtype, data, mask=np.ma.nomask, mask_corners=True, 
                data_folded=None, check_folding=True, dtype=float, copy=True, 
                fill_value=np.nan, keep_mask=True, shrink=True, 
                pop_ids=None, extrap_x=None):
        data = np.asanyarray(data)

        if mask is np.ma.nomask:
            mask = np.ma.make_mask_none(data.shape)

        subarr = np.ma.masked_array(data, mask=mask, dtype=dtype, copy=copy,
                                       fill_value=fill_value, keep_mask=True, 
                                       shrink=True)
        subarr = subarr.view(subtype)

        if hasattr(data, 'folded'):
            if data_folded is None or data_folded == data.folded:
                subarr.folded = data.folded
            elif data_folded != data.folded:
                raise ValueError('Data does not have same folding status as '
                                 'was called for in Spectrum constructor.')
        elif data_folded is not None:
            subarr.folded = data_folded
        else:
            subarr.folded = False

        # Check that if we're declaring that the input data is folded, it
        # actually is, and the mask reflects this.
        if data_folded:
            total_samples = np.sum(subarr.sample_sizes)
            total_per_entry = subarr._total_per_entry()
            # Which entries are nonsense in the folded fs.
            where_folded_out = total_per_entry > int(total_samples/2)
            if check_folding\
               and not np.all(subarr.data[where_folded_out] == 0):
                print('Creating Spectrum with data_folded = True, but '
                            'data has non-zero values in entries which are '
                            'nonsensical for a folded Spectrum.')
            if check_folding\
               and not np.all(subarr.mask[where_folded_out]):
                print('Creating Spectrum with data_folded = True, but '
                            'mask is not True for all entries which are '
                            'nonsensical for a folded Spectrum.')

        if hasattr(data, 'pop_ids'):
            if pop_ids is None or pop_ids == data.pop_ids:
                subarr.pop_ids = data.pop_ids
            elif pop_ids != data.pop_ids:
                print('Changing population labels in construction of new '
                            'Spectrum.')
                if len(pop_ids) != subarr.ndim:
                    raise ValueError('pop_ids must be of length equal to '
                                     'dimensionality of Spectrum.')
                subarr.pop_ids = pop_ids
        else:
            if pop_ids is not None and len(pop_ids) != subarr.ndim:
                raise ValueError('pop_ids must be of length equal to '
                                 'dimensionality of Spectrum.')
            subarr.pop_ids = pop_ids

        if mask_corners:
            subarr.mask_corners()

        subarr.extrap_x = extrap_x

        return subarr

    # See http://www.scipy.org/Subclasses for information on the
    # __array_finalize__ and __array_wrap__ methods. I had to do some debugging
    # myself to discover that I also needed _update_from.
    # Also, see http://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    # Also, see http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    #
    # We need these methods to ensure extra attributes get copied along when
    # we do arithmetic on the FS.
    def __array_finalize__(self, obj):
        if obj is None: 
            return
        np.ma.masked_array.__array_finalize__(self, obj)
        self.folded = getattr(obj, 'folded', 'unspecified')
        self.pop_ids = getattr(obj, 'pop_ids', None)
        self.extrap_x = getattr(obj, 'extrap_x', None)
    def __array_wrap__(self, obj, context=None):
        result = obj.view(type(self))
        result = np.ma.masked_array.__array_wrap__(self, obj, 
                                                      context=context)
        result.folded = self.folded
        result.pop_ids = self.pop_ids
        result.extrap_x = self.extrap_x
        return result
    def _update_from(self, obj):
        np.ma.masked_array._update_from(self, obj)
        if hasattr(obj, 'folded'):
            self.folded = obj.folded
        if hasattr(obj, 'pop_ids'):
            self.pop_ids = obj.pop_ids
        if hasattr(obj, 'extrap_x'):
            self.extrap_x = obj.extrap_x
    # masked_array has priority 15.
    __array_priority__ = 20

    def __repr__(self):
        return 'Spectrum(%s, folded=%s, pop_ids=%s)'\
                % (str(self), str(self.folded), str(self.pop_ids))

    def mask_corners(self):
        """
        Mask the 'seen in 0 samples' and 'seen in all samples' entries.
        """
        self.mask.flat[0] = self.mask.flat[-1] = True

    def unmask_all(self):
        """
        Unmask all values.
        """
        self.mask[[slice(None)]*self.Npop] = False

    def _get_sample_sizes(self):
        return np.asarray(self.shape) - 1
    sample_sizes = property(_get_sample_sizes)

    def _get_Npop(self):
        return self.ndim
    Npop = property(_get_Npop)

    def _ensure_dimension(self, Npop):
        """
        Ensure that fs has Npop dimensions.
        """
        if not self.Npop == Npop:
            raise ValueError('Only compatible with %id spectra.' % Npop)

    # Make from_file a static method, so we can use it without an instance.
    @staticmethod
    def from_file(fname, mask_corners=True, return_comments=False):
        """
        Read frequency spectrum from file.

        fname: String with file name to read from. If it ends in .gz, gzip
               compression is assumed.
        mask_corners: If True, mask the 'absent in all samples' and 'fixed in
                      all samples' entries.
        return_comments: If true, the return value is (fs, comments), where
                         comments is a list of strings containing the comments
                         from the file (without #'s).

        See to_file method for details on the file format.
        """
        if fname.endswith('.gz'):
            fid = gzip.open(fname, 'rb')
        else:
            fid = open(fname, 'r')

        line = fid.readline()
        # Strip out the comments
        comments = []
        while line.startswith('#'):
            comments.append(line[1:].strip())
            line = fid.readline()

        # Read the shape of the data
        shape_spl = line.split()
        if 'folded' not in shape_spl and 'unfolded' not in shape_spl:
            # This case handles the old file format
            shape = tuple([int(d) for d in shape_spl])
            folded = False
            pop_ids = None
        else:
            # This case handles the new file format
            shape,next_ii = [int(shape_spl[0])], 1
            while shape_spl[next_ii] not in ['folded', 'unfolded']:
                shape.append(int(shape_spl[next_ii]))
                next_ii += 1
            folded = (shape_spl[next_ii] == 'folded')
            # Are there population labels in the file?
            if len(shape_spl) > next_ii + 1:
                pop_ids = line.split('"')[1::2]
            else:
                pop_ids = None

        data = np.fromstring(fid.readline().strip(), 
                                count=np.product(shape), sep=' ')
        # fromfile returns a 1-d array. Reshape it to the proper form.
        data = data.reshape(*shape)

        maskline = fid.readline().strip()
        if not maskline:
            # The old file format didn't have a line for the mask
            mask = None
        else:
            # This case handles the new file format
            mask = np.fromstring(maskline, 
                                    count=np.product(shape), sep=' ')
            mask = mask.reshape(*shape)

        fs = Spectrum(data, mask, mask_corners, data_folded=folded,
                      pop_ids=pop_ids)

        fid.close()
        if not return_comments:
            return fs
        else:
            return fs,comments

    fromfile = from_file

    def to_file(self, fname, precision=16, comment_lines = [], 
                foldmaskinfo=True):
        """
        Write frequency spectrum to file.

        fname: File name to write to.  If string ends in .gz, file will be saved
               with gzip compression.
        precision: precision with which to write out entries of the SFS. (They
                   are formated via %.<p>g, where <p> is the precision.)
        comment lines: list of strings to be used as comment lines in the header
                       of the output file.
        foldmaskinfo: If False, folding and mask and population label
                      information will not be saved. This conforms to the file
                      format for dadi versions prior to 1.3.0.

        The file format is:
            # Any number of comment lines beginning with a '#'
            A single line containing N integers giving the dimensions of the fs
              array. So this line would be '5 5 3' for an SFS that was 5x5x3.
              (That would be 4x4x2 *samples*.)
            On the *same line*, the string 'folded' or 'unfolded' denoting the
              folding status of the array
            On the *same line*, optional strings each containing the population
              labels in quotes separated by spaces, e.g. "pop 1" "pop 2"
            A single line giving the array elements. The order of elements is 
              e.g.: fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1] ...
            A single line giving the elements of the mask in the same order as
              the data line. '1' indicates masked, '0' indicates unmasked.
        """
        # Open the file object.
        if fname.endswith('.gz'):
            fid = gzip.open(fname, 'wb')
        else:
            fid = open(fname, 'w')

        # Write comments
        for line in comment_lines:
            fid.write('# ')
            fid.write(line.strip())
            fid.write('\n')

        # Write out the shape of the fs
        for elem in self.data.shape:
            fid.write('%i ' % elem)

        if foldmaskinfo:
            if not self.folded:
                fid.write('unfolded')
            else:
                fid.write('folded')
            if self.pop_ids is not None:
                for label in self.pop_ids:
                    fid.write(' "%s"' % label)

        fid.write('\n')

        # Write the data to the file. The obnoxious ravel call is to
        # ensure compatibility with old version that used self.data.tofile.
        np.savetxt(fid, [self.data.ravel()], delimiter=' ',
                      fmt='%%.%ig' % precision)

        if foldmaskinfo:
            # Write the mask to the file
            np.savetxt(fid, [np.asarray(self.mask, int).ravel()],
                          delimiter=' ', fmt='%d')

        fid.close()

    tofile = to_file


    def project(self, ns):
        """
        Project to smaller sample size.

        ns: Sample sizes for new spectrum.
        """
        if len(ns) != self.Npop:
            raise ValueError('Requested sample sizes not of same dimension '
                             'as spectrum. Perhaps you need to marginalize '
                             'over some populations first?')
        if np.any(np.asarray(ns) > np.asarray(self.sample_sizes)):
            raise ValueError('Cannot project to a sample size greater than '
                             'original. Original size is %s and requested size '
                             'is %s.' % (self.sample_sizes, ns))

        original_folded = self.folded
        # If we started with an folded Spectrum, we need to unfold before
        # projecting.
        if original_folded:
            output = self.unfold()
        else:
            output = self.copy()

        # Iterate over each axis, applying the projection.
        for axis,proj in enumerate(ns):
            if proj != self.sample_sizes[axis]:
                output = output._project_one_axis(proj, axis)

        output.pop_ids = self.pop_ids
        output.extrap_x = self.extrap_x

        # Return folded or unfolded as original.
        if original_folded:
            return output.fold()
        else:
            return output

    def _project_one_axis(self, n, axis=0):
        """
        Project along a single axis.
        """
        # This gets a little tricky with fancy indexing to make it work
        # for fs with arbitrary number of dimensions.
        if n > self.sample_sizes[axis]:
            raise ValueError('Cannot project to a sample size greater than '
                             'original. Called sizes were from %s to %s.' 
                             % (self.sample_sizes[axis], n))

        newshape = list(self.shape)
        newshape[axis] = n+1
        # Create a new empty fs that we'll fill in below.
        pfs = Spectrum(np.zeros(newshape), mask_corners=False)

        # Set up for our fancy indexes. These slices are currently like
        # [:,:,...]
        from_slice = [slice(None) for ii in range(self.Npop)]
        to_slice = [slice(None) for ii in range(self.Npop)]
        proj_slice = [nuax for ii in range(self.Npop)]

        proj_from = self.sample_sizes[axis]
        # For each possible number of hits.
        for hits in range(proj_from+1):
            # Adjust the slice in the array we're projecting from.
            from_slice[axis] = slice(hits, hits+1)
            # These are the least and most possible hits we could have in the
            #  projected fs.
            least, most = max(n - (proj_from - hits), 0), min(hits,n)
            to_slice[axis] = slice(least, most+1)
            # The projection weights.
            proj = _cached_projection(n, proj_from, hits)
            proj_slice[axis] = slice(least, most+1)
            # Do the multiplications
            pfs.data[tuple(to_slice)] += self.data[tuple(from_slice)] * proj[tuple(proj_slice)]
            pfs.mask[tuple(to_slice)] = np.logical_or(pfs.mask[tuple(to_slice)],
                                                         self.mask[tuple(from_slice)])

        return pfs

    def marginalize(self, over, mask_corners=True):
        """
        Reduced dimensionality spectrum summing over some populations.

        over: sequence of axes to sum over. For example (0,2) will sum over
              populations 0 and 2.
        mask_corners: If True, the typical corners of the resulting fs will be
                      masked
        """
        original_folded = self.folded
        # If we started with an folded Spectrum, we need to unfold before
        # marginalizing.
        if original_folded:
            output = self.unfold()
        else:
            output = self.copy()

        orig_mask = output.mask.copy()
        orig_mask.flat[0] = orig_mask.flat[-1] = False
        if np.any(orig_mask):
            print('Marginalizing a Spectrum with internal masked values. '
                        'This may not be a well-defined operation.')

        # Do the marginalization
        for axis in sorted(over)[::-1]:
            output = output.sum(axis=axis)
        pop_ids = None
        if self.pop_ids is not None:
            pop_ids = list(self.pop_ids)
            for axis in sorted(over)[::-1]:
                del pop_ids[axis]
        output.folded = False
        output.pop_ids = pop_ids
        output.extrap_x = self.extrap_x

        if mask_corners:
            output.mask_corners()

        # Return folded or unfolded as original.
        if original_folded:
            return output.fold()
        else:
            return output


    def _counts_per_entry(self):
        """
        Counts per population for each entry in the fs.
        """
        ind = np.indices(self.shape)
        # Transpose the first access to the last, so ind[ii,jj,kk] = [ii,jj,kk]
        ind = ind.transpose(list(range(1,self.Npop+1))+[0])
        return ind

    def _total_per_entry(self):
        """
        Total derived alleles for each entry in the fs.
        """
        return np.sum(self._counts_per_entry(), axis=-1)

    def log(self):
        """
        Return the natural logarithm of the entries of the frequency spectrum.

        Only necessary because np.ma.log now fails to propagate extra
        attributes after numpy 1.10.
        """
        logfs = np.ma.log(self)
        logfs.folded = self.folded
        logfs.pop_ids = self.pop_ids
        logfs.extrap_x = self.extrap_x
        return logfs

    def reorder_pops(self, neworder):
        """
        Get Spectrum with populations in new order

        Returns new Spectrum with same number of populations, but in a different order

        neworder: Integer list defining new order of populations, indexing the orginal
                  populations from 1. Must contain all integers from 1 to number of pops.
        """
        if sorted(neworder) != [_+1 for _ in range(self.ndim)]:
            raise(ValueError("neworder argument misspecified"))
        newaxes = [_-1 for _ in neworder]
        fs = self.transpose(newaxes)
        if self.pop_ids:
            fs.pop_ids = [self.pop_ids[_] for _ in newaxes]

        return fs

    def fold(self):
        """
        Folded frequency spectrum
    
        The folded fs assumes that information on which allele is ancestral or
        derived is unavailable. Thus the fs is in terms of minor allele 
        frequency.  Note that this makes the fs into a "triangular" array.
    
        Note that if a masked cell is folded into non-masked cell, the
        destination cell is masked as well.

        Note also that folding is not done in-place. The return value is a new
        Spectrum object.
        """
        if self.folded:
            raise ValueError('Input Spectrum is already folded.')

        # How many samples total do we have? The folded fs can only contain
        # entries up to total_samples/2 (rounded down).
        total_samples = np.sum(self.sample_sizes)

        total_per_entry = self._total_per_entry()
    
        # Here's where we calculate which entries are nonsense in the folded fs.
        where_folded_out = total_per_entry > int(total_samples/2)
    
        original_mask = self.mask
        # Here we create a mask that masks any values that were masked in
        # the original fs (or folded onto by a masked value).
        final_mask = np.logical_or(original_mask, 
                                      reverse_array(original_mask))
        
        # To do the actual folding, we take those entries that would be folded
        # out, reverse the array along all axes, and add them back to the
        # original fs.
        reversed = reverse_array(np.where(where_folded_out, self, 0))
        folded = np.ma.masked_array(self.data + reversed)
        folded.data[where_folded_out] = 0
    
        # Deal with those entries where assignment of the minor allele is
        # ambiguous.
        where_ambiguous = (total_per_entry == total_samples/2.)
        ambiguous = np.where(where_ambiguous, self, 0)
        folded += -0.5*ambiguous + 0.5*reverse_array(ambiguous)
    
        # Mask out the remains of the folding operation.
        final_mask = np.logical_or(final_mask, where_folded_out)

        outfs = Spectrum(folded, mask=final_mask, data_folded=True,
                         pop_ids=self.pop_ids)
        outfs.extrap_x = self.extrap_x
        return outfs

    def unfold(self):
        """
        Unfolded frequency spectrum
    
        It is assumed that each state of a SNP is equally likely to be
        ancestral.

        Note also that unfolding is not done in-place. The return value is a new
        Spectrum object.
        """
        if not self.folded:
            raise ValueError('Input Spectrum is not folded.')

        # Unfolding the data is easy.
        reversed_data = reverse_array(self.data)
        newdata = (self.data + reversed_data)/2.

        # Unfolding the mask is trickier. We want to preserve masking of entries
        # that were masked in the original Spectrum.
        # Which entries in the original Spectrum were masked solely because
        # they are incompatible with a folded Spectrum?
        total_samples = np.sum(self.sample_sizes)
        total_per_entry = self._total_per_entry()
        where_folded_out = total_per_entry > int(total_samples/2)

        newmask = np.logical_xor(self.mask, where_folded_out)
        newmask = np.logical_or(newmask, reverse_array(newmask))
    
        outfs = Spectrum(newdata, mask=newmask, data_folded=False, 
                         pop_ids=self.pop_ids)
        outfs.extrap_x = self.extrap_x
        return outfs

    def fixed_size_sample(self, nsamples, only_nonmasked=False):
        """
        Generate a resampled fs from the current one.

        nsamples: Number of samples to include in the new FS.
        only_nonmasked: If True, only SNPs from non-masked will be resampled. 
                        Otherwise, all SNPs will be used.
        """
        flat = self.flatten()
        if only_nonmasked:
            pvals = flat.data/flat.sum()
            pvals[flat.mask] = 0
        else:
            pvals = flat.data/flat.data.sum()
    
        sample = np.random.multinomial(int(nsamples), pvals)
        sample = sample.reshape(self.shape)
    
        return dadi.Spectrum(sample, mask=self.mask, pop_ids=self.pop_ids)

    def sample(self):
        """
        Generate a Poisson-sampled fs from the current one.

        Note: Entries where the current fs is masked will be masked in the
              output sampled fs.
        """
        import scipy.stats
        # These are entries where the sampling has no meaning, b/c fs is masked.
        bad_entries = self.mask
        # We convert to a 1-d array for passing into the sampler
        means = self.ravel().copy()
        # Filter out those bad entries.
        means[bad_entries.ravel()] = 1
        # Sample
        samp = scipy.stats.distributions.poisson.rvs(means, size=len(means))
        # Replace bad entries with zero
        samp[bad_entries.ravel()] = 0
        # Convert back to a properly shaped array
        samp = samp.reshape(self.shape)
        # Convert to a fs and mask the bad entries
        samp = Spectrum(samp, mask=self.mask, data_folded=self.folded,
                        pop_ids = self.pop_ids)
        return samp


    def Fst(self):
        """
        Wright's Fst between the populations represented in the fs.
    
        This estimate of Fst assumes random mating, because we don't have
        heterozygote frequencies in the fs.
    
        Calculation is by the method of Weir and Cockerham _Evolution_ 38:1358
        (1984).  For a single SNP, the relevant formula is at the top of page
        1363. To combine results between SNPs, we use the weighted average
        indicated by equation 10.
        """
        # This gets a little obscure because we want to be able to work with
        # spectra of arbitrary dimension.
    
        # First quantities from page 1360
        r = self.Npop
        ns = self.sample_sizes
        nbar = np.mean(ns)
        nsum = np.sum(ns)
        nc = (nsum - np.sum(ns**2)/nsum)/(r-1)
    
        # counts_per_pop is an r+1 dimensional array, where the last axis simply
        # records the indices of the entry. 
        # For example, counts_per_pop[4,19,8] = [4,19,8]
        counts_per_pop = np.indices(self.shape)
        counts_per_pop = np.transpose(counts_per_pop, axes=list(range(1,r+1))+[0])
    
        # The last axis of ptwiddle is now the relative frequency of SNPs in
        # that bin in each of the populations.
        ptwiddle = 1.*counts_per_pop/ns
    
        # Note that pbar is of the same shape as fs...
        pbar = np.sum(ns*ptwiddle, axis=-1)/nsum
    
        # We need to use 'this_slice' to get the proper aligment between
        # ptwiddle and pbar.
        this_slice = [slice(None)]*r + [np.newaxis]
        s2 = np.sum(ns * (ptwiddle - pbar[tuple(this_slice)])**2, axis=-1)/((r-1)*nbar)
    
        # Note that this 'a' differs from equation 2, because we've used
        # equation 3 and b = 0 to solve for hbar.
        a = nbar/nc * (s2 - 1/(2*nbar-1) * (pbar*(1-pbar) - (r-1)/r*s2))
        d = 2*nbar/(2*nbar-1) * (pbar*(1-pbar) - (r-1)/r*s2)
    
        # The weighted sum over loci.
        asum = (self * a).sum()
        dsum = (self * d).sum()
    
        return asum/(asum+dsum)

    def S(self):
        """
        Segregating sites.
        """
        oldmask = self.mask.copy()
        self.mask_corners()
        S = self.sum()
        self.mask = oldmask
        return S
    
    def Watterson_theta(self):
        """
        Watterson's estimator of theta.
    
        Note that is only sensible for 1-dimensional spectra.
        """
        if self.Npop != 1:
            raise ValueError("Only defined on a one-dimensional fs.")
    
        n = self.sample_sizes[0]
        S = self.S()
        an = np.sum(1./np.arange(1,n))
    
        return S/an
    
    def theta_L(self):
        """
        theta_L as defined by Zeng et al. "Statistical Tests for Detecting
        Positive Selection by Utilizing High-Frequency Variants" (2006)
        Genetics
    
        Note that is only sensible for 1-dimensional spectra.
        """
        if self.Npop != 1:
            raise ValueError("Only defined on a one-dimensional fs.")
    
        n = self.sample_sizes[0]
        return np.sum(np.arange(1,n)*self[1:n])/(n-1)

    def Zengs_E(self):
        """
        Zeng et al.'s E statistic.

        From Zeng et al. "Statistical Tests for Detecting Positive Selection by
        Utilizing High-Frequency Variants" (2006) Genetics
        """
        num = self.theta_L() - self.Watterson_theta()

        n = self.sample_sizes[0]

        # See after Eq. 3
        an = np.sum(1./np.arange(1,n))
        # See after Eq. 9
        bn = np.sum(1./np.arange(1,n)**2)
        s = self.S()

        # See immediately after Eq. 12
        theta = self.Watterson_theta()
        theta_sq = s*(s-1.)/(an**2 + bn)

        # Eq. 14
        var = (n/(2.*(n-1.)) - 1./an) * theta\
                + (bn/an**2 + 2.*(n/(n-1.))**2 * bn - 2*(n*bn-n+1.)/((n-1.)*an)
                   - (3.*n+1.)/(n-1.)) * theta_sq

        return num/np.sqrt(var)
    
    def pi(self):
        r"""
        Estimated expected number of pairwise differences between two
        chromosomes in the population.
    
        Note that this estimate includes a factor of sample_size/(sample_size-1)
        to make E(\hat{pi}) = theta.
        """
        if self.ndim != 1:
            raise ValueError("Only defined for a one-dimensional SFS.")
    
        n = self.sample_sizes[0]
        # sample frequencies p 
        p = np.arange(0,n+1,dtype=float)/n
        # This expression derives from Gillespie's _Population_Genetics:_A
        # _Concise_Guide_, 2nd edition, section 2.6.
        return n/(n-1.) * 2*np.ma.sum(self*p*(1-p))
    
    def Tajima_D(self):
        """
        Tajima's D.
    
        Following Gillespie "Population Genetics: A Concise Guide" pg. 45
        """
        if not self.Npop == 1:
            raise ValueError("Only defined on a one-dimensional SFS.")
    
        S = self.S()
    
        n = 1.*self.sample_sizes[0]
        pihat = self.pi()
        theta = self.Watterson_theta()
    
        a1 = np.sum(1./np.arange(1,n))
        a2 = np.sum(1./np.arange(1,n)**2)
        b1 = (n+1)/(3*(n-1))
        b2 = 2*(n**2 + n + 3)/(9*n * (n-1))
        c1 = b1 - 1./a1
        c2 = b2 - (n+2)/(a1*n) + a2/a1**2
    
        C = np.sqrt((c1/a1)*S + c2/(a1**2 + a2) * S*(S-1))
    
        return (pihat - theta)/C


    @staticmethod
    def from_data_dict(data_dict, pop_ids, projections, mask_corners=True,
                       polarized=True):
        """
        Spectrum from a dictionary of polymorphisms.

        pop_ids: list of which populations to make fs for.
        projections: list of sample sizes to project down to for each
                     population.
        mask_corners: If True (default), the 'observed in none' and 'observed 
                      in all' entries of the FS will be masked.
        polarized: If True, the data are assumed to be correctly polarized by 
                   `outgroup_allele'. SNPs in which the 'outgroup_allele'
                   information is missing or '-' or not concordant with the
                   segregating alleles will be ignored.
                   If False, any 'outgroup_allele' info present is ignored,
                   and the returned spectrum is folded.

        The data dictionary should be organized as:
            {snp_id:{'segregating': ['A','T'],
                     'calls': {'YRI': (23,3),
                                'CEU': (7,3)
                                },
                     'outgroup_allele': 'T'
                    }
            }
        The 'calls' entry gives the successful calls in each population, in the
        order that the alleles are specified in 'segregating'.
        Non-diallelic polymorphisms are skipped.
        """
        cd = Spectrum.count_data_dict(data_dict, pop_ids)
        fs = Spectrum._from_count_dict(cd, projections, polarized, pop_ids, 
                mask_corners=mask_corners)
        return fs

    @staticmethod
    def count_data_dict(data_dict, pop_ids):
        """
        Summarize data in data_dict by mapping SNP configurations to counts.
    
        data_dict: data_dict formatted as in Misc.make_data_dict
        pop_ids: IDs of populations to collect data for.
    
        Returns a dictionary with keys (successful_calls, derived_calls,
        polarized) mapping to counts of SNPs. Here successful_calls is a tuple
        with the number of good calls per population, derived_calls is a tuple
        of derived calls per pop, and polarized indicates whether that SNP was
        polarized using an ancestral state.
        """
        count_dict = collections.defaultdict(int)
        for snp_info in data_dict.values():
            # Skip SNPs that aren't biallelic.
            if len(snp_info['segregating']) != 2:
                continue
    
            allele1,allele2 = snp_info['segregating']
            if 'outgroup_allele' in snp_info and snp_info['outgroup_allele'] != '-'\
                and snp_info['outgroup_allele'] in snp_info['segregating']:
                outgroup_allele = snp_info['outgroup_allele']
                this_snp_polarized = True
            else:
                outgroup_allele = allele1
                this_snp_polarized = False
    
            # Extract the allele calls for each population.
            allele1_calls = [snp_info['calls'][pop][0] for pop in pop_ids]
            allele2_calls = [snp_info['calls'][pop][1] for pop in pop_ids]
            # How many chromosomes did we call successfully in each population?
            successful_calls = [a1+a2 for (a1,a2)
                                in zip(allele1_calls, allele2_calls)]
    
            # Which allele is derived (different from outgroup)?
            if allele1 == outgroup_allele:
                derived_calls = allele2_calls
            elif allele2 == outgroup_allele:
                derived_calls = allele1_calls
    
            # Update count_dict
            count_dict[tuple(successful_calls),tuple(derived_calls),
                       this_snp_polarized] += 1
        return count_dict

    @staticmethod
    def _from_count_dict(count_dict, projections, polarized=True, pop_ids=None,
            mask_corners=False):
        """
        Frequency spectrum from data mapping SNP configurations to counts.

        count_dict: Result of Misc.count_data_dict
        projections: List of sample sizes to project down to for each
                     population.
        polarized: If True, only include SNPs that count_dict marks as
                   polarized.
                   If False, include all SNPs and fold resulting Spectrum.
        pop_ids: Optional list of strings containing the population labels.
        mask_corners: If True (default), the 'observed in none' and 'observed 
                      in all' entries of the FS will be masked.
        """
        # create slices for projection calculation
        slices = [[np.newaxis] * len(projections) for ii in
                  range(len(projections))]
        for ii in range(len(projections)):
            slices[ii][ii] = slice(None,None,None)
        # Convert to tuples to avoid numpy error
        slices = [tuple(_) for _ in slices]

        fs_total = Spectrum(np.zeros(np.array(projections)+1),
                                 pop_ids=pop_ids, mask_corners=mask_corners)
        for (called_by_pop, derived_by_pop, this_snp_polarized), count\
                in count_dict.items():
            if polarized and not this_snp_polarized:
                continue
            pop_contribs = []
            iter = zip(projections, called_by_pop, derived_by_pop)
            for pop_ii, (p_to, p_from, hits) in enumerate(iter):
                contrib = _cached_projection(p_to,p_from,hits)[slices[pop_ii]]
                pop_contribs.append(contrib)
            fs_proj = pop_contribs[0]
            for contrib in pop_contribs[1:]:
                fs_proj = fs_proj*contrib

            # create slices for adding projected fs to overall fs
            fs_total += count * fs_proj
        if polarized:
            return fs_total
        else:
            return fs_total.fold()

    
    # The code below ensures that when I do arithmetic with Spectrum objects,
    # it is not done between a folded and an unfolded array. If it is, I raise
    # a ValueError.

    # While I'm at it, I'm also fixing the annoying behavior that if a1 and a2
    # are masked arrays, and a3 = a1 + a2. Then wherever a1 or a2 was masked,
    # a3.data ends up with the a1.data values, rather than a1.data + a2.data.
    # Note that this fix doesn't work for operation by np.ma.exp and 
    # np.ma.log. Guess I can't have everything.

    # I'm using exec here to avoid copy-pasting a dozen boiler-plate functions.
    # The calls to check_folding_equal ensure that we don't try to combine
    # folded and unfolded Spectrum objects.

    # I set check_folding = False in the constructor because it raises useless
    # warnings when, for example, I do (model + 1). 

    # These functions also ensure that the pop_ids and extrap_x attributes
    # get properly copied over.

    # This is pretty advanced Python voodoo, so don't fret if you don't
    # understand it at first glance. :-)
    for method in ['__add__','__radd__','__sub__','__rsub__','__mul__',
                   '__rmul__','__div__','__rdiv__','__truediv__','__rtruediv__',
                   '__floordiv__','__rfloordiv__','__rpow__','__pow__']:
        exec("""
def %(method)s(self, other):
    self._check_other_folding(other)
    if isinstance(other, np.ma.masked_array):
        newdata = self.data.%(method)s (other.data)
        newmask = np.ma.mask_or(self.mask, other.mask)
    else:
        newdata = self.data.%(method)s (other)
        newmask = self.mask
    newpop_ids = self.pop_ids
    if hasattr(other, 'pop_ids'):
        if other.pop_ids is None:
            newpop_ids = self.pop_ids
        elif self.pop_ids is None:
            newpop_ids = other.pop_ids
        elif other.pop_ids != self.pop_ids:
            print('Arithmetic between Spectra with different pop_ids. '
                        'Resulting pop_id may not be correct.')
    if hasattr(other, 'extrap_x') and self.extrap_x != other.extrap_x:
        extrap_x = None
    else:
        extrap_x = self.extrap_x
    outfs = self.__class__.__new__(self.__class__, newdata, newmask, 
                                   mask_corners=False, data_folded=self.folded,
                                   check_folding=False, pop_ids=newpop_ids,
                                   extrap_x=extrap_x)
    return outfs
""" % {'method':method})

    # Methods that modify the Spectrum in-place.
    for method in ['__iadd__','__isub__','__imul__','__idiv__',
                   '__itruediv__','__ifloordiv__','__ipow__']:
        exec("""
def %(method)s(self, other):
    self._check_other_folding(other)
    if isinstance(other, np.ma.masked_array):
        self.data.%(method)s (other.data)
        self.mask = np.ma.mask_or(self.mask, other.mask)
    else:
        self.data.%(method)s (other)
    if hasattr(other, 'pop_ids') and other.pop_ids is not None\
             and other.pop_ids != self.pop_ids:
        print('Arithmetic between Spectra with different pop_ids. '
                    'Resulting pop_id may not be correct.')
    if hasattr(other, 'extrap_x') and self.extrap_x != other.extrap_x:
        self.extrap_x = None
    return self
""" % {'method':method})

    def _check_other_folding(self, other):
        """
        Ensure other Spectrum has same .folded status
        """
        if isinstance(other, self.__class__)\
           and other.folded != self.folded:
            raise ValueError('Cannot operate with a folded Spectrum and an '
                             'unfolded one.')

