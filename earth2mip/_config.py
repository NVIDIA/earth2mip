from typing import List
from pydantic import BaseSettings
from earth2mip import schema


class Settings(BaseSettings):
    # not needed anymore
    # AFNO_26_WEIGHTS: Optional[str] = None
    # AFNO_26_MEAN: str
    # AFNO_26_SCALE: str

    # only used in earth2mip.diagnostics
    # TODO add defaults (maybe scope in that module)
    MEAN: str = ""
    SCALE: str = ""

    # Key configurations
    ERA5_HDF5_34: str = ""
    ERA5_HDF5_73: str = ""
    MODEL_REGISTRY: str = ""
    LOCAL_CACHE: str = ""

    # used for scoring (score-ifs.py, inference-medium-range)
    TIME_MEAN: str = ""
    TIME_MEAN_73: str = ""

    # used in score-ifs.py
    # TODO refactor to a command line argument of that script
    IFS_ROOT: str = None

    # only used in test suite
    # TODO add a default option.
    TEST_DIAGNOSTICS: List[str] = ()

    # where to store regridding files
    MAP_FILES: str = ""

    class Config:
        env_file = ".env"

    def get_data_root(self, channel_set: schema.ChannelSet) -> str:
        if channel_set == schema.ChannelSet.var34:
            val = self.ERA5_HDF5_34
            if not val:
                raise ValueError(
                    "Please configure ERA5_HDF5_34 to point to the 34 channel data."  # noqa
                )
            return val
        elif channel_set == schema.ChannelSet.var73:
            val = self.ERA5_HDF5_73
            if not val:
                raise ValueError("Please configure ERA5_HDF5_73.")
        else:
            raise NotImplementedError(channel_set)

        return val

    def get_time_mean(self, channel_set: schema.ChannelSet) -> str:
        return {
            schema.ChannelSet.var34: self.TIME_MEAN,
            schema.ChannelSet.var73: self.TIME_MEAN_73,
        }[channel_set]
