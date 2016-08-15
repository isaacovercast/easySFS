# easySFS
Convert VCF to dadi/fastsimcoal style SFS for demographic analysis

## Install & Run
The script assumes you have matplotlib and dadi installed.
* Clone this repo
* `cd easySFS`
* `chmod 777 easySFS`
* `./easySFS`

## General workflow
Converting VCF to SFS is a 2 step process. The first step is to run a preview to identify the values for projecting down each population. The second step is to actually do the conversion specifying the the projection values. It looks like this:

`./easySFS -i input.vcf -p pops_file.txt --preview`

Which will output something like this:
```
Pop1
2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  
49.0    62.0    57.0    51.0    42.0    34.0    29.0    21.0    13.0    8.0 8.0 5.0 2.0 2.0 1.0 

Pop2
2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  
63.0    89.0    102.0   108.0   104.0   88.0    72.0    64.0    52.0    46.0    36.0    32.0    27.0    23.0    22.0    19.0    18.0    17.0    16.0    17.0    15.0    14.0    12.0    12.0    12.0    12.0    11.0    11.0    10.0    10.0    11.0    10.0    10.0    7.0 6.0 6.0 6.0 6.0 4.0 4.0 4.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
```
Each column is the number of samples in the projection and the number of segregating sites at that projection value.
The dadi manual recommends maximinzing the number of segregating sites, but at the same time if you have lots of missing data then you might have to balance # of segregating sites against # of samples to avoid downsampling too far.

Next run the script with the values for projecting for each population, like this:

`./easySFS -i input.vcf -p pops_file.txt --proj 12,20`

## Outputs
If you specify the `-o` flag you can pass in an output directory which will be created, otherwise output files are written to the default directory `output`. There will be two directories created here `dadi` and `fastsimcoal2`

## Running example files




## Usage
You can get usage info any time by: `./easySFS.py`
```
usage: easySFS.py [-h] [-a] -i VCF_NAME -p POPULATIONS [--proj PROJECTIONS]
                  [--preview] [-o OUTDIR] [--ploidy PLOIDY] [--prefix PREFIX]
                  [--GQ GQUAL] [-f] [-v]

optional arguments:
  -h, --help          show this help message and exit
  -a                  Keep all snps (default == False)
  -i VCF_NAME         name of the VCF input file being converted
  -p POPULATIONS      Input file containing population assignments per
                      individual
  --proj PROJECTIONS  List of values for projecting populations down to
                      different sample sizes
  --preview           Preview the number of segragating sites per population
                      for different projection values.
  -o OUTDIR           Directory to write output SFS to
  --ploidy PLOIDY     Specify ploidy. Default is 2. Only other option is 1 for
                      haploid.
  --prefix PREFIX     Prefix for all output SFS files names.
  --GQ GQUAL          minimum genotype quality tolerated
  -f                  Force overwriting directories and existing files.
  -v                  Set verbosity. Dump tons of info to the screen
```
