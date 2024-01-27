# trimet-gtfs-parser
Understanding ridership, routes, frequencies, and more

Three main files, gtfs.py deals with directly parsing data from
the GTFS feed and should more or less work for any transit agency.
trimet.py deals with TriMet specific data, and mostly was used to 
fetch data for maps, tables, and charts in (this) ArcGIS Story Map

The ridership parser was written in R, since the pdf text extraction
is better. Creates a bunch of fixed width files that were in turn
read by trimet.py to get ridership by stop, and by route.

