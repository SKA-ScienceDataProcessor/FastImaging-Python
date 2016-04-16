# fastimg_extract_lsm

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

## Installation
Installed as part of the `fastimgproto` Python package.

## Usage
After installing the `fastimgproto` package
*fastimg_extract_lsm* can then be run from the command line, e.g.
    
    fastimg_extract_lsm --help

Then e.g. to extract all sources within 2 degrees of the RA/Dec coordinates (25, -45), run:

    fastimg_extract_lsm -- 25 -45 2
    
(The `--` separator prevents the interpreter from confusing `-45` with an options flag, omitting it will result in 'Error: no such option: -4'.)
    
Optionally you can specify an output filename ('csv' format, but currently uses tab-delimiters as it's easier to read):

    fastimg_extract_lsm -- 25 -45 2 foobar.csv
