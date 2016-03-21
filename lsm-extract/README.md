# lsm-extract

Provides model / simulated data for the local sky model.

This is currently implemented using the Sydney University Molonglo Sky Survey (SUMSS) catalog. The 'extract_lsm.py' script loads the SUMSS catalog from file and outputs a list of all sources within the specified circular region.

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


    
