# trimet specific data

import os
from typing import Optional, Union

import pandas as pd
import geopandas as gpd
from shapely import Point

from gtfs import GTFS

class TriMet(GTFS):
    # container for TriMet specific (i.e. ridership) data

    def __init__(
        self, 
        path: os.PathLike,
        ridership_path: os.PathLike
    ) -> None:
        super().__init__(path)

        self.ridership_path = ridership_path
        self.ridership_file = os.path.basename(ridership_path)
        self.ridership_folder = os.path.join(
            os.path.dirname(ridership_path),
            self.ridership_file.replace(".pdf","")
        )

        if not os.path.isdir(self.ridership_folder):
            os.system(f"Rscript {os.path.dirname(__file__)}/stop_ridership_parser.r")
            if not os.path.isdir(self.ridership_folder):
                raise ValueError("Improper file name, R script failed to create as well")
            
        self.ridership = self._read_ridership_folder()

    def _read_ridership_folder(self) -> pd.DataFrame:
        """read set of fwf from ridership folder"""
        data_list = []
        for file_name in os.listdir(self.ridership_folder):
            file_path = os.path.join(self.ridership_folder, file_name)
            data = pd.read_fwf(file_path, colspecs="infer", header=None)
            # find unnamed bits between stop location and location id
            cols = data.columns.to_list()

            # sometimes, multiple stop columns are infered for stop locations
            stop_loc_cols = data.columns[:len(data.columns)-8]
            if len(stop_loc_cols) > 1:
                data[stop_loc_cols] = data[stop_loc_cols].fillna("").astype(str)
                data[0] = data[stop_loc_cols].agg(" ".join, axis=1)
                data.drop(stop_loc_cols[1:], axis=1, inplace=True)
            
            columns = [
                "stop_location","stop_id","direction","position",
                "ons","offs","total","sep","monthly_lifts"
            ]
            data.columns = columns
            try:
                data["ons"] = data["ons"].str.replace(",","")
            except AttributeError:
                pass
            try:
                data["offs"] = data["offs"].str.replace(",","")
            except AttributeError:
                pass
            try:
                data["total"] = data["total"].str.replace(",","")
            except AttributeError:
                pass
            data.drop("sep",axis=1,inplace=True)
            data_list.append(data)

        df = pd.concat(data_list)
        df.dropna(how="all",inplace=True)
        df["stop_id"] = df["stop_id"].astype(int)
        df["ons"] = pd.to_numeric(df["ons"])
        df["offs"] = pd.to_numeric(df["offs"])
        df["total"] = pd.to_numeric(df["total"])
        return df
    
    def line_ridership(
        self, 
        route_id: Optional[str] = None,
        day_type: str = "weekday",
        add_stop_location: bool = False,
        add_stop_geom: bool = False,
        add_headways_all_hours: bool = False
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Merge line data with ridership data, day_type should match
        the day of the ridership data
        """
        if self.stop_times is None or self.stop_times.empty:
            self.stop_times = pd.read_csv(os.path.join(self.path, "stop_times.txt"))
        
        cal = self._calendar_from_day_type(day_type)
        cal["cal_days_active"] = (
            (pd.to_datetime(cal["end_date"],format="%Y%m%d") 
             - pd.to_datetime(cal["start_date"],format="%Y%m%d"))
        ).dt.days
        # filter to one service id that has the most days active
        # this should be a "typical" service day
        cal = cal[cal["cal_days_active"] == cal["cal_days_active"].max()]
        cal.drop_duplicates(subset="service_id",inplace=True,ignore_index=True)
        trips = self.trips.merge(
            cal[["service_id"]],
            on="service_id",
            how="inner"
        )
        
        stop_times = self.stop_times.merge(
            trips,
            on="trip_id",
            how="inner",
            suffixes=["","_trips"]
        )
        # group count of trips by stop id, direction id, maybe route
        # (this matters for shared stops, like MAX, not so much for buses)
        # can't group by direction id, since some stops serve buses in different directions
        if route_id is not None:
            stop_info = stop_times.groupby(
                by=["stop_id","route_id"],
                as_index=False
            )[["trip_id"]].count()
            stop_info.rename({"trip_id":f"{day_type}_trips"},inplace=True,axis=1)
            # now, we need to filter to just the stop ids served by our route
            just_our_route = stop_info[stop_info["route_id"] == route_id]
            stop_info = stop_info[stop_info["stop_id"].isin(just_our_route["stop_id"].unique())]
            # re aggregate by stop/direction, sum trips. this is to include all trips
            # at each stop, regardless of what route serves it to accurately depict
            # per-bus level ridership along route
            # some bus stops are served by different routes in different direction_ids,
            # so we need to aggregate at the stop_id level for buses, but potentially
            # stop and direction for MAX?
            stop_info = stop_info.groupby(
                by=["stop_id"],
                as_index=False
            ).agg(
                {
                    f"{day_type}_trips":"sum",
                    "route_id":"nunique"
                }
            )
            stop_info.rename({"route_id":"routes_served_at_stop"},axis=1,inplace=True)
        else:
            stop_info = stop_times.groupby(
                by=["stop_id"],
                as_index=False
            ).agg(
                {
                    "trip_id":"count",
                    "route_id":"nunique"
                }
            )
            stop_info.rename(
                {
                    "trip_id":f"{day_type}_trips",
                    "route_id":"routes_served_at_stop"
                },
                axis=1,
                inplace=True
            )
        # merge ridership data
        stop_info = stop_info.merge(
            self.ridership,
            on="stop_id",
            how="inner"
        )
        stop_info["boards_per_bus"] = stop_info["total"].astype(float) / stop_info[f"{day_type}_trips"]
        
        if not add_stop_location and not add_stop_geom:
            return stop_info
        
        if add_stop_location:
            stop_info["stop_id"] = stop_info["stop_id"].astype(str)
            stop_info = stop_info.merge(
                self.stops[["stop_id","stop_lat","stop_lon"]], 
                on="stop_id"
            )
            stop_info["stop_point"] = [
                Point(xy) for xy in zip(stop_info["stop_lon"],stop_info["stop_lat"])
            ]
            stop_info = gpd.GeoDataFrame(data=stop_info, geometry=stop_info["stop_point"])
            stop_info.drop("stop_point",axis=1,inplace=True)
            if not add_headways_all_hours:
                return stop_info
    
        if add_stop_geom:
            # get headway geography for the route
            headway_geom = self.typical_headway(
                time_start="06:00:00",
                time_end="07:00:00",
                route_id=route_id,
                day_type=day_type,
                with_geom=True
            )
            stop_info["stop_id"] = stop_info["stop_id"].astype(str)
            stop_info = stop_info.merge(
                headway_geom[["stop_id","direction_id","schedule_headway","geometry"]],
                on="stop_id",
                how="left"
            )
            stop_info = gpd.GeoDataFrame(
                data=stop_info,
                geometry=stop_info["geometry"]
            )
            stop_info.rename({"schedule_headway":"schedule_headway_6"},inplace=True,axis=1)
        # fetch other hours, if passed
        if not add_headways_all_hours:
            return stop_info
        
        # add every hour of day 7 to 23
        hours_to_add = [
            "07:00:00","08:00:00",
            "09:00:00","10:00:00","11:00:00",
            "12:00:00","13:00:00","14:00:00",
            "15:00:00","16:00:00","17:00:00",
            "18:00:00","19:00:00","20:00:00",
            "21:00:00","22:00:00","23:00:00",
            "24:00:00"
        ]
        for hour in hours_to_add:
            time_start_num = self.hour_num_from_str(hour)
            time_end_num = time_start_num + 1
            hour_df = self.typical_headway(
                route_id=route_id,
                time_start=time_start_num,
                time_end=time_end_num,
                day_type=day_type
            )
            hour_df.rename(
                {"schedule_headway":f"schedule_headway_{int(time_start_num)}"},
                axis=1,
                inplace=True
            )
            merge_cols = ["stop_id",f"schedule_headway_{int(time_start_num)}"]
            hour_df["stop_id"] = hour_df["stop_id"].astype(str)
            stop_info = stop_info.merge(
                hour_df[merge_cols],
                on="stop_id",
                how="inner"
            )
            stop_info.drop_duplicates(subset="stop_id", inplace=True)
            

        return stop_info

def main():
    # more data
    tm = TriMet(
        path="/Users/andrewmurp/Documents/python/gtfs/_data/gtfs_trimet_latest",
        ridership_path="/Users/andrewmurp/Documents/python/gtfs/_data/stop_level_ridership_spring_2023.pdf"
    )
    df = tm.line_ridership(
        route_id="17",
        add_stop_geom=True,
        add_headways_all_hours=True
    )

    df.to_file(
        os.path.join(
            os.path.dirname(__file__),
            "_data",
            "output_stop_ridership",
            "line_17_full_data_weekdays_v3.geojson"
        )
    )
    df[df.columns[:-1]].to_csv(
        os.path.join(
            os.path.dirname(__file__),
            "_data",
            "output_stop_ridership",
            "line_17_ridership_headway_only.csv"
        )
    )
    return df

def test():
    # fetch some data about stop ids, etc. for arcgis
    tm = TriMet(
        path="/Users/andrewmurp/Documents/python/gtfs/_data/gtfs_trimet_latest",
        ridership_path="/Users/andrewmurp/Documents/python/gtfs/_data/stop_level_ridership_spring_2023.pdf"
    )
    df = tm.line_ridership()
    df_70 = tm.line_ridership(route_id="75",add_stop_location=True)

    output_path = os.path.join(
        os.path.dirname(__file__),
        "_data",
        "output_stop_ridership"
    )
    os.makedirs(output_path, exist_ok=True)

    df.to_csv(os.path.join(output_path, "ridership_data_overall.csv"))
    df_70.to_file(os.path.join(output_path, "stop_locs_75.geojson"))

def main2():
    """get median riders per bus by line"""
    tm = TriMet(
        path="/Users/andrewmurp/Documents/python/gtfs/_data/gtfs_trimet_latest",
        ridership_path="/Users/andrewmurp/Documents/python/gtfs/_data/stop_level_ridership_spring_2023.pdf"
    )
    all_data = []
    for rid in tm.routes["route_id"].unique():
        rlr = tm.line_ridership(route_id=rid)
        rlr["route_id"] = rid
        all_data.append(rlr)

    df = pd.concat(all_data)
    print(df)
    output_path = os.path.join(
        os.path.dirname(__file__),
        "_data",
        "output_stop_ridership"
    )
    gf = df.groupby(by=["route_id"]).agg(
        {
            "boards_per_bus":"median"
        }
    )
    gf.to_csv(
        os.path.join(output_path, "ridership_data_by_route.csv")
    )

if __name__ == "__main__":
    df = test()

