**This code is old and not maintained.  I strongly recommend that you use the newer adapt-vqe code which is pinned on my github.**

	OF_VQE.py is a fairly self-explanatory UCCSD code.  If you want to use a molecule not specified, you will need to input the geometry alongside the others.  Most of the options are for the benefit of computing dissociation curves, so you can just use the defaults for general calculations.

You can save some hassle by cd'ing to the working directory (OF_VQE by default) and using

    pip install -r /path/to/requirements.txt

You will also need a working Psi4 install, instructions available here:

    http://www.psicode.org/psi4manual/1.2/external.html

If you have any issues with the code, feel free to e-mail me at

    hrgrimsl@vt.edu

Flags:

-l:  Log output file; defaults to app.log but can be anything.

-m:  Determines minimum level of a seriousness to be displayed in the log.  Defaults to DEBUG, but the threshold can be raised to INFO to get only the final error from FCI.

-rw: Access log file in mode 'a' or 'w'; defaults to w, i.e. obliterates the log file rather than appending to it.

-s:  Random seed for shuffling operators.  Defaults to 111596.

-p:  Protocol for arranging operators.  Defaults to a random arrangement.  It is also possible to choose:
    
    increasing_comms - Increasing [e^A,H] values

    decreasing_comms - Decreasing [e^A,H] values

    increasing_unexp_comms - Increasing [A, H] values
    
    decreasing_comms - Decreasing [A, H] values

-sys: System of interest; defaults to water.  Must be listed in the code.

-d:  Dissociation parameter; defaults to 0, i.e. equilibrium bond distance.  Causes the H's to move d angstroms away from the 'central' atom.

-cc:  Defaults to False.  When on, the program will use classical CCSD amplitudes as starting parameters for the UCCSD calculation.

-b:  Defaults to minimal STO-3G basis.  Includes all the normal Psi4 orbitals.  Likely choices include:

    sto-3g - minimal basis
    
    3-21g - small split valence
    
    3-21g* - small split valence w/ polarization
    
    6-31g - larger split valence

    6-31g* - larger split valence w/ polarization

    cc-pvdz - correlation-consistent basis

    (Note that anything larger than STO-3G will fail for all but the smallest systems!)

-f:  Filter out terms by their contribution to the Hamiltonian?  Defaults to no, is probably more important in the pqrs scheme.

-c:  Default ijab.

    ijab - allow only the unitary versions of traditional single and double excitations, i.e. only occ->nocc transitions are permitted.  This will generally be substantially faster and only marginally less accurate.

    pqrs - allow all non-degenerate excitations which can possibly non-zero, including nocc->nocc, occ->occ, and spin-flips.  This is crazy expensive, but you'll have so many degrees of freedom that America will be jealous, so you'll probably get a slightly better answer.
