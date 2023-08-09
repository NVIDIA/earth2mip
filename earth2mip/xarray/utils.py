import xarray
import cupy


def to_cupy(ds):
    """Convert a dataset or datarray to use cupy

    assumes that dask is not used
    """
    if isinstance(ds, xarray.DataArray):
        # TODO make this name unique
        name = "23vd89a"
        return to_cupy(ds.to_dataset(name=name))[name]

    variables = {}
    for v in ds.variables:
        try:
            arr = cupy.asarray(ds[v])
        except ValueError as e:
            raise ValueError(
                f"{v} cannot be converted to cupy. Original exception: {e}."
            )
        else:
            variables[v] = xarray.Variable(ds[v].dims, arr)

    return ds._replace_with_new_dims(
        variables, coord_names=list(ds.coords), indexes=ds.indexes
    )


def to_np(arr):
    return xarray.DataArray(
        arr.data.get(), coords=arr.coords, dims=arr.dims, attrs=arr.attrs
    )


def concat_dict(d, key_names=(), concat_dim="key"):
    """concat a dictionary of xarray objects"""
    arrays = []
    for key, arr in d.items():
        coords = {}
        for name, value in zip(key_names, key):
            coords[name] = xarray.Variable([concat_dim], [value])
        arr = arr.expand_dims(concat_dim)
        arr = arr.assign_coords(coords)
        arrays.append(arr)
    return xarray.concat(arrays, dim=concat_dim)
