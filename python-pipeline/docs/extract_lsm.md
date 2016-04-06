# extract_lsm.py

A quick'n'dirty script to produce an arbitrarily formatted 'local-sky-model'
by extracting a subset of a larger sky-catalog from a local catalog-file.

Currently configured to load data from the Sydney University Molonglo Sky
Survey (SUMSS) catalog, cf.
http://www.astrop.physics.usyd.edu.au/sumsscat/description.html

Only circular extraction regions are implemented thus far.

We make use of Pandas dataframes together with the RA/DEC boxing algorithm of
*[Gray (2006)](http://research.microsoft.com/pubs/64524/tr-2006-52.pdf)*
to narrow down the source list to a subset that may be in the requested field
of view. Final angular distance comparisons are then made using
astropy.coordinates functionality. Extraction of a region of 1-degree radius
takes a few seconds.

## Usage
You will need a local copy of the SUMSS catalog file from 
http://www.astrop.physics.usyd.edu.au/sumsscat/sumsscat.Mar-11-2008.gz (cf *download_sumsscat.sh*).

You will also need to install the Python libraries listed in *requirements.txt*.

*extract_lsm.py* can then be run as a script, e.g.
    
    ./extract_lsm.py --help

Then e.g. to extract all sources within 2 degrees of the RA/Dec coordinates (25, -45), run:

    ./extract_lsm.py -- 25 -45 2
    
(The `--` separator prevents the interpreter from confusing `-45` with an options flag, omitting it will result in 'Error: no such option: -4'.)
    
Optionally you can specify an output filename ('csv' format, but currently uses tab-delimiters as it's easier to read):

    ./extract_lsm.py -- 25 -45 2 foobar.csv

By default, 'extract_lsm.py' looks for the SUMSS catalog in your current directory, but you can specify a path if necessary:

    ./extract_lsm.py --catalog-file /some/other/dir/sumsscat.Mar-11-2008.gz -- 25 -45 2
