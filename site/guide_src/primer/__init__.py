"""Content modules for the Beginner's Guide (site/docs/primer/).

Its own package so a driver imports exactly ONE guide's figures and
worked examples per process — see guides.py. The shared machinery
(_style, lensing, cosmo, registry) is imported from the parent dir, which
is sys.path[0] when the tools run.
"""
