Spectacle: Spectral learning for Annotating Chromatin Labels and Epigenomes

Copyright (C) 2013-2015 Jimin Song

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========
Spectacle
=========

This software implements a spectral learning algorithm for hidden Markov models for epigenomic data.
Please see our paper for further details.

Song, J and Chen, K. C. Spectacle: fast chromatin state annotation using spectral learning.
Genome Biology, 16:33, 2015.
http://genomebiology.com/2015/16/1/33

Spectacle is written in Java and based on the ``ChromHMM`` code. It has been tested on Linux and
Windows. It extends ``LearnModel`` and ``MakeSegmentation`` among the top level commands of ``ChromHMM``
(http://compbio.mit.edu/ChromHMM) by adding a spectral learning algorithm for Hidden Markov Model
parameter estimation. Other than ```LearnModel`` and ``MakeSegmentation``, all other commands are run as
described in the ``ChromHMM`` manual (http://compbio.mit.edu/ChromHMM/ChromHMM_manual.pdf).


Installation
============

- Install Java 1.5 or later if not already installed.

- Download Spectacle::

    Download https://github.com/jiminsong/Spectacle/archive/master.zip
    # If you're on Ubuntu or Mac
    git clone https://github.com/jiminsong/Spectacle.git

|

EXAMPLE
=======

Here is an example of how to run ``Spectacle.jar`` with ChIP-seq data from ENCODE.

Prepare the dataset
-------------------

- Download ChIP-seq data from `ENCODE website`_ and put it in the folder ``SAMPLEDATA_HG19``.
  ``hg19_cellmarkfiletable.txt`` contains the names of the files that were downloaded.

- Binarize Bed files::

    java -jar Spectacle.jar BinarizeBed CHROMSIZES/hg19.txt SAMPLEDATA_HG19 hg19_cellmarkfiletable.txt SAMPLEDATA_HG19

  ``hg19_inputfilelist.txt`` contains the filenames of the binarized data.

.. _`ENCODE website`: http://www.broadinstitute.org/~anshul/projects/encode/rawdata/mapped/jan2011/noMultiMapTagAlign/


Spectral Learning
-----------------

``Spectacle.jar`` estimates Hidden Markov Model parameters using a spectral learning algorithm as described in our `paper`_.


We also reimplemented the spectral learning algorithm using Python sparse matrix libraries to allow
users to analyze a large number of chromatin marks. The Python module makes use of the sparsity of
the observed pairs matrix for the SVD computation. The module is implemented in Python using the
SciPy library (Python version 2.7.6 and SciPy version 0.13.3) for the SVD computation. The Python
module is wrapped with the Spectacle Java code for other tasks- e.g. reading datasets, computing the
likelihood, and assigning chromatin states to genomic segments by posterior decoding algorithm etc.

We recommend using the Python module if the number of chromatin marks you would like
to use is large.

Spectral Learning without the Python module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- run on whole genome::

    java -mx1200M -jar Spectacle.jar LearnModel -nobrowser -noenrich -f "hg19_inputfilelist.txt" -i spectral -lambda 1 -comb -p 4 SAMPLEDATA_HG19 OUTPUTSAMPLE_HG19 20 hg19

- run on one chromosome::

    java -mx1200M -jar Spectacle.jar LearnModel -nobrowser -noenrich -f "hg19_inputfilelist1.txt" -i spectral_chr18 -lambda 1 -comb -p 4 SAMPLEDATA_HG19 OUTPUTSAMPLE_HG19 20 hg19


Spectral Learning with the Python module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- run on whole genome::

    java -mx1200M -jar Spectacle.jar LearnModel -nobed -nobrowser -noenrich -f "hg19_inputfilelist.txt" -i spectral -lambda 1 -p 4 -computesamplemomentonly SAMPLEDATA_HG19 OUTPUTSAMPLE_HG19 20 hg19
    python Spectacle_python.py OUTPUTSAMPLE_HG19 spectral 20 8
    java -jar Spectacle.jar MakeSegmentation -f "hg19_inputfilelist.txt" -i spectral -comb "OUTPUTSAMPLE_HG19/model_comb_20_spectral.txt" SAMPLEDATA_HG19 OUTPUTSAMPLE_HG19

- run on one chromosome::

    java -mx1200M -jar Spectacle.jar LearnModel -nobed -nobrowser -noenrich -f "hg19_inputfilelist1.txt" -i spectral_chr18 -lambda 1 -p 4 -computesamplemomentonly SAMPLEDATA_HG19 OUTPUTSAMPLE_HG19 20 hg19
    python Spectacle_python.py OUTPUTSAMPLE_HG19 spectral_chr18 20 8
    java -jar Spectacle.jar MakeSegmentation -f "hg19_inputfilelist1.txt" -i spectral_chr18_comb -comb "OUTPUTSAMPLE_HG19/model_comb_20_spectral_chr18.txt" SAMPLEDATA_HG19 OUTPUTSAMPLE_HG19


Options for LearnModel
----------------------

We provide several options in addition to the options of ``ChromHMM`` for the *LearnModel* command.

- ``-chromhmm`` : If this flag is set, ``Spectacle`` is run exactly the same as ``ChromHMM``
  as described in the ChromHMM manual. Otherwise, it is run using the spectral learning algorithm.

- ``-lambda`` *lambdavalue* : The Lambda parameter in the data smoothing step is set as *lambdavalue*.
  *lambdavalue*=0.95 is the default.

- ``-spectralrandom`` : The emission matrix is estimated as described in Appendix C in Hsu et al 2012.
  Without this option, the emission matrix is estimated using major observations as described
  in our `paper`_.

- ``-comb`` (``noindep`` in the previous versions) : Do not use assumption that all histone marks
  are conditionally independent given the chromatin state, and consider observations as
  combinations of marks. There are no EM iterations by default. If EM iterations are run,
  observations are not considered as combinations of marks anymore.

- ``-p`` *maxprocessors* : A model is trained using multiple processors in parallel.

- ``-r`` *nmaxiterations* : Number of EM iterations. *nmaxiterations*=0 (i.e. no iteration) is
  the default, in which parameters are not changed but the likelihood is just computed.

- ``-init load_comb`` : This loads the parameters specified in ``modelinitialfile`` which is
  in format of output file of the python module ``Spectacle_python.py``. By loading the output file
  of ``Spectacle_python.py``, ``-comb`` is automatically set.

- ``computesamplemomentonly`` : This reads histone mark datasets and computes sample moments for
  spectral learning. Sample moments are written in a file, which is an input file for the python
  module ``Spectacle_python.py``.

Options for Spectral_python.py
------------------------------

``Spectacle_python.py`` runs as follows::

    python Spectral_python.py file_directory fileID num_states num_marks (min_occurrence_observation)

where *min_occurrence_observation* specifies the minimum number of occurrences of the observation
in the genome and the default number is 1.

.. _`paper`: http://genomebiology.com/2015/16/1/33


References
==========

- Ernst J, Kellis M. ChromHMM: automating chromatin state discovery and characterization. Nature Methods, 9:215-216, 2012.
- Hsu D, Kakade SM, Zhang T. A spectral algorithm for learning hidden Markov models. Journal of Computer and System Sciences, 78:1460-1480, 2012.
- Song, J and Chen, K. C. Spectacle: fast chromatin state annotation using spectral learning. Genome Biology, 16:33, 2015.
