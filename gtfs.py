# coding: utf - 8

"""
    Author: Andrew Lindstrom
    Date: 2024-01-22
    Purpose:
        GTFS parser and functions related
"""

import os
import zipfile
import logging
import datetime as datetime
from typing import Union

import pandas as pd
import numpy as np
import geopandas as gpd

import shapely.ops
from shapely import Point, LineString, MultiPoint

import matplotlib.pyplot as plt

# see this link https://stackoverflow.com/questions/34754777/shapely-split-linestrings-at-intersections-with-other-linestrings
def cut_linestring_at_points(linestring, points):
    non_empty_pts = [pt for pt in points if not pt.is_empty]
    return shapely.ops.split(linestring, MultiPoint(non_empty_pts))

class GTFS(object):
    # container for GTFS data
    def __init__(
        self,
        path: os.PathLike
    ) -> None:
        if os.path.isfile(path):
            # try unzipping
            if not zipfile.is_zipfile(path):
                raise TypeError("Path must be a path or a zipfile")
            with zipfile.ZipFile(path,"r") as zf:
                output_path = path.replace(".zip","")
                zf.extractall(output_path)
                self.path = output_path
        elif os.path.exists(path):
            self.path = path
        else:
            raise ValueError("Path must be an existing path or zipfile")
        
    
        self.routes = pd.read_csv(os.path.join(self.path, "routes.txt"))
        self.trips = pd.read_csv(os.path.join(self.path, "trips.txt"))
        self.calendar = pd.read_csv(os.path.join(self.path, "calendar.txt"))
        self.stops = pd.read_csv(os.path.join(self.path, "stops.txt"))
        # forward declare, leave null until referenced though
        self.stop_times = pd.DataFrame()
        self.shapes = pd.DataFrame()
        self.shape_geometry = gpd.GeoDataFrame()
        self.calendar_dates = pd.DataFrame()

    def _calendar_from_day_type(
        self,
        day_type: str
    ) -> pd.DataFrame:
        # pass in day_type, return calendar df of those service_ids and s/e dates
        # handle case when calendar is 0 for all days... why do you do this TriMet
        all_cols = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        if (self.calendar[all_cols] == 0).all().all():
            # we need to make self.calender from calendar dates..
            if self.calendar_dates is None or self.calendar_dates.empty:
                self.calendar_dates = pd.read_csv(os.path.join(self.path,"calendar_dates.txt"))
            
            self.calendar_dates["date"] = pd.to_datetime(
                self.calendar_dates["date"],
                format="%Y%m%d"
            )
            self.calendar_dates["day_of_week"] = self.calendar_dates["date"].dt.strftime("%A").str.lower()

            for dow in self.calendar_dates["day_of_week"].unique():
                self.calendar_dates[dow] = (self.calendar_dates["day_of_week"] == dow).astype(int)

            gf = self.calendar_dates.groupby(
                by=["service_id"],
                as_index=False
            ).agg(
                {
                    "monday":"sum",
                    "tuesday":"sum",
                    "wednesday":"sum",
                    "thursday":"sum",
                    "friday":"sum",
                    "saturday":"sum",
                    "sunday":"sum",
                    "date":("min","max"),
                }
            )
            gf.columns = [
                "service_id","monday","tuesday","wednesday",
                "thursday","friday","saturday","sunday",
                "start_date","end_date"
            ]
            gf[all_cols] = np.where(
                gf[all_cols] > 0,
                1,
                0
            )
            made_calender = True
        else:
            made_calender = False
        
        if day_type == "weekday":
            ref_cols = ["monday","tuesday","wednesday","thursday","friday"]
        elif day_type == "weekend":
            ref_cols = ["saturday","sunday"]
        else:
            ref_cols = [day_type]

        if not made_calender:
            return self.calendar[(self.calendar[ref_cols] == 1).all(axis=1)]
        else:
            return gf[(gf[ref_cols] == 1).all(axis=1)]
    
    @staticmethod
    def hour_num_from_str(
        hour: Union[str, int, float],
        fmt: str = "hh:mm:ss"
    ) -> float:
        if not isinstance(hour, str):
            return hour
        else:
            return (
                int(hour.split(":")[0]) 
                + (float(hour.split(":")[1]) / 60)
                + (float(hour.split(":")[2]) / 3600)
            )
    
    @staticmethod
    def _split_line_with_point(line, splitter):
        """Split a LineString with a Point"""
        # taken from https://github.com/shapely/shapely/blob/4fcf83523dbe96fcc51fe12b6826a6ecdc4b2a66/shapely/ops.py#L450

        #
        assert(isinstance(splitter, Point))
        # if a list of lines is passed, use the one closer to the point
        if isinstance(line, list):
            d1 = line[0].distance(splitter)
            d2 = line[1].distance(splitter)
            if d1 < d2:
                line = line[0]
            else:
                line = line[1]
        assert(isinstance(line, LineString))

        # check if point is in the interior of the line
        """if not line.relate_pattern(splitter, '0********'):
            # point not on line interior --> return collection with single identity line
            # (REASONING: Returning a list with the input line reference and creating a
            # GeometryCollection at the general split function prevents unnecessary copying
            # of linestrings in multipoint splitting function)
            return [line]"""
        if line.coords[0] == splitter.coords[0]:
            # if line is a closed ring the previous test doesn't behave as desired
            return [line]

        # point is on line, get the distance from the first point on line
        distance_on_line = line.project(splitter)
        coords = list(line.coords)
        # split the line at the point and create two new lines
        current_position = 0.0
        for i in range(len(coords)-1):
            point1 = coords[i]
            point2 = coords[i+1]
            dx = point1[0] - point2[0]
            dy = point1[1] - point2[1]
            segment_length = (dx ** 2 + dy ** 2) ** 0.5
            current_position += segment_length
            if distance_on_line == current_position:
                # splitter is exactly on a vertex
                # sometimes fails if line constructed would have one pt
                try:
                    return [
                        LineString(coords[:i+2]),
                        LineString(coords[i+1:])
                    ]
                except Exception:
                    return [
                        LineString((splitter, coords[i+1])),
                        LineString((splitter, coords[i+1]))
                    ]
            elif distance_on_line < current_position:
                # splitter is between two vertices
                # sometimes fails if line constructed would have one pt
                try:
                    return [
                        LineString(coords[:i+1] + [splitter.coords[0]]),
                        LineString([splitter.coords[0]] + coords[i+1:])
                    ]
                except Exception:
                    return [
                        LineString((splitter, coords[i+1])),
                        LineString((splitter, coords[i+1]))
                    ]
        return [line]

    def headway_by_time(
        self,
        time: str,
        day_type: str = "weekday",
        level: str = "stop"
    ) -> gpd.GeoDataFrame:
        """create dataframe for all routes at a given time
        
        Args:
            time (required): string for local time in 24 hour format, though >24 hour will be 1 am up to ~1:30 
                (this is based on TriMet - I don't know how other agencies handle 24 hour periods)

            day_type (optional): weekday, weekend, or any day of week
            defaults to weekday

            level (optional): stop or route. defaults to stop
        """
        time_num = self.hour_num_from_str(time)
        if self.stop_times is None or self.stop_times.empty:
            logging.info("Loading stop times, may be slow")
            self.stop_times = pd.read_csv(os.path.join(self.path, "stop_times.txt"))
        
        cal = self._calendar_from_day_type(day_type)
        
        # total days active for to weight by in calendar?
        cal["cal_days_active"] = (
            (pd.to_datetime(cal["end_date"],format="%Y%m%d") 
             - pd.to_datetime(cal["start_date"],format="%Y%m%d"))
        ).dt.days

        trips = self.trips.merge(
            cal[["service_id","cal_days_active"]],
            on="service_id",
            how="inner"
        )
        stop_times = self.stop_times.merge(
            trips,
            on="trip_id",
            how="inner",
            suffixes=["","_trips"]
        )
        stop_times["departure_time"] = stop_times["departure_time"].apply(self.hour_num_from_str)
        stop_times["arrival_time"] = stop_times["arrival_time"].apply(self.hour_num_from_str)

        # only want next bus after and before
        after = stop_times[stop_times["departure_time"] > time_num]
        before = stop_times[stop_times["departure_time"] <= time_num]

        # filter to only the first after, or final before
        after_agg = after.groupby(
            by=["route_id", "service_id","direction_id","stop_id"], as_index=False
        )[["departure_time"]].min()
        before_agg = before.groupby(
            by=["route_id","service_id","direction_id","stop_id"], as_index=False
        )[["departure_time"]].max()

        after_agg["next_bus_in"] = after_agg["departure_time"] - time_num
        before_agg["last_bus_was"] = time_num - before_agg["departure_time"]

        stop_times = stop_times.merge(
            after_agg,
            on=["route_id","service_id","direction_id","departure_time","stop_id"],
            how="left"
        )
        stop_times = stop_times.merge(
            before_agg,
            on=["route_id", "service_id","direction_id","departure_time","stop_id"],
            how="left"
        )

        stop_times = stop_times[
            (stop_times["next_bus_in"].notnull())
            | (stop_times["last_bus_was"].notnull())
        ]
        # TODO - add something to distinguish branches
        # in minutes!
        stop_times["headway"] = 60*stop_times["next_bus_in"].fillna(stop_times["last_bus_was"])

        if level == "stop":
            stop_times = stop_times.merge(self.routes,on="route_id")
            output_cols = [
                "route_id","route_short_name","route_long_name","departure_time",
                "shape_id","trip_type","trip_id",
                "direction_id","stop_headsign","stop_id",
                "service_id","cal_days_active",
                "headway"
            ]
            return stop_times[output_cols]
        
        # aggregate at route/direction level, weighted sum with days active in calendar
        # since that seems like a good idea
        stop_times["headway_wt"] = stop_times["headway"] * stop_times["cal_days_active"]
        agged_stops = stop_times.groupby(
            by=["route_id","direction_id"],
            as_index=False
        )[["headway_wt","cal_days_active"]].sum()
        agged_stops["headway"] = agged_stops["headway_wt"] / agged_stops["cal_days_active"]
        # everything is doubled counted since we have two trips (the final before, and first after)
        agged_stops["headway"] = agged_stops["headway"] * 2

        # clean output, add a few columns
        agged_stops = agged_stops.merge(
            self.routes[["route_id","route_short_name","route_long_name"]],
            on="route_id"
        )
        return agged_stops
    
    def headway_geography(
        self,
        time: str = "08:00:00",
        filter_by_route: str = None,
        day_type: str = "weekday"
    ) -> gpd.GeoDataFrame:
        """Create a geodataframe with a bunch of lines and headways for each route

        the goal here is to have a headway and linestring
        """
        
        if self.shapes is None or self.shapes.empty:
            self.shapes = pd.read_csv(os.path.join(self.path, "shapes.txt"))

        if self.shape_geometry is None or self.shape_geometry.empty:
            # construct geodataframe from shape.txt file
            geometry = [
                Point(xy) for xy in zip(self.shapes["shape_pt_lon"],self.shapes["shape_pt_lat"])
            ]
            gdf = gpd.GeoDataFrame(self.shapes, geometry=geometry)

            gdf = gdf.groupby(by="shape_id", as_index=False)["geometry"].apply(lambda x: LineString(x.tolist()))

            self.shape_geometry = gdf
        else:
            gdf = self.shape_geometry

        stop_headways = self.headway_by_time(time=time, day_type=day_type)

        #stop_headways.to_csv(os.path.join(os.path.dirname(__file__),"stopheadways.csv"))
        if filter_by_route is not None:
            stop_headways = stop_headways[stop_headways["route_id"].astype(str) == str(filter_by_route)]
        # key thing to note - each stop should be a point on the route - like a literal point in
        # the line string (or it is on TriMet!)
        # we want to show the headway at the time for each subsection of the route, so need to cut route
        # on each pair of stops in turn

        stop_headways["stop_id"] = stop_headways["stop_id"].astype(str)
        stop_headways = stop_headways.merge(
            self.stops[["stop_id","stop_lat","stop_lon"]],
            on="stop_id",
            how="inner"
        )
        # now, we need to get the next point in sequence
        stop_headways.sort_values(
            by=["route_id","trip_id","departure_time"],
            inplace=True
        )
        stop_headways["next_stop_lat"] = stop_headways.groupby(
            by=["trip_id"]
        )["stop_lat"].shift(-1)
        stop_headways["next_stop_lon"] = stop_headways.groupby(
            by=["trip_id"]
        )["stop_lon"].shift(-1)

        # create gdf
        stop_headways["stop_point"] = [
            Point(xy) for xy in zip(stop_headways["stop_lon"],stop_headways["stop_lat"])
        ]
        stop_headways["next_stop_point"] = [
            Point(xy) for xy in zip(stop_headways["next_stop_lon"],stop_headways["next_stop_lat"])
        ]
        gdf = stop_headways.merge(
            gdf,
            how="inner",
            on="shape_id",
            suffixes=["_stops",""]
        )
        # remove end points (with no next stop) - TODO - fix later, not sure this makes sense to do or not
        gdf = gdf[gdf["next_stop_lon"].notnull()]

        gdf["sub_geometry"] = gdf.apply(
            lambda x: self._split_line_with_point(x["geometry"],x["stop_point"]),
            axis=1
        )
        gdf["sub_geometry"] = gdf.apply(
            lambda x: self._split_line_with_point(x["sub_geometry"],x["next_stop_point"]),
            axis=1
        )
        # final step, select the shorter of the two lines
        gdf["line_between_stops"] = gdf["sub_geometry"].apply(
            lambda x: x[0] if x[0].length < x[1].length else x[1]
        )

        gdf["old_geometry"] = gdf["geometry"].copy()
        gdf["geometry"] = gdf["line_between_stops"]

        return gdf

    def typical_headway(
        self,
        route_id: str = "17",
        time_start: str = "08:00:00",
        time_end: str = "09:00:00",
        time_steps: int = 6,
        with_geom: bool = False,
        day_type: str = "weekday"
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Get typical headway for route between two times
        """
        time_start_num = self.hour_num_from_str(time_start)
        time_end_num = self.hour_num_from_str(time_end)
        time_step = (time_end_num - time_start_num) / time_steps

        times = [time_start_num + time_step*n for n in range(time_steps + 1)]

        df_list = []
        for time in times:
            if with_geom:
                df = self.headway_geography(
                    time=time, 
                    filter_by_route=route_id,
                    day_type=day_type
                )
            else:
                df = self.headway_by_time(time=time, level="stop")
                df = df[df["route_id"] == route_id]
            df_list.append(df)

        df = pd.concat(df_list)
        
        gf = df.groupby(
            by=["stop_id","route_id","direction_id",],
            as_index=False
        )[["headway"]].mean()

        # the mean headway is always half of the literal headway of a bus over time
        # if a bus comes every 30 minutes to a stop, then over a 30 minute span
        # the bus will be coming in 0,1,2,3 ... 28, 29, 30 minutes - a mean of 15
        gf["schedule_headway"] = gf["headway"] * 2
        gf.rename({"headway":"mean_headway"},axis=1,inplace=True)

        # with geom re-add at stop id
        if with_geom:
            gf = gf.merge(
                df[["stop_id","geometry"]].drop_duplicates(),
                on="stop_id"
            )
            gf = gpd.GeoDataFrame(data=gf, geometry=gf["geometry"])
        return gf

def main():
    """Get some data!"""
    path_to_file = "/Users/andrewmurp/Documents/python/gtfs/_data/gtfs_trimet_latest"

    trimet = GTFS(path_to_file)

    # number 17, data every 10 minutes, 6 am to 11 pm
    hours = [
        "06:00:00","07:00:00","08:00:00",
        "09:00:00","10:00:00","11:00:00",
        "12:00:00","13:00:00","14:00:00",
        "15:00:00","16:00:00","17:00:00",
        "18:00:00","19:00:00","20:00:00",
        "21:00:00","22:00:00","23:00:00",
        "24:00:00"
    ]
    for idx in range(len(hours)-1):
        start = hours[idx]
        end = hours[idx+1]
        df = trimet.typical_headway(
            route_id="17",
            time_start=start,
            time_end=end,
            time_steps=6,
            with_geom=True,
            day_type="saturday"
        )
        print(f"Ran data for {start} to {end}")
        df.to_file(
            os.path.join(
                os.path.dirname(__file__),
                "route_17_weekend",
                f"typical_headway_17_{start[:2]}_{end[:2]}.geojson"
            )
        )


if __name__ == "__main__":
    main()
