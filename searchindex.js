Search.setIndex({"docnames": ["api", "clis", "concepts", "index", "plugin", "troubleshooting", "tutorial"], "filenames": ["api.rst", "clis.rst", "concepts.rst", "index.rst", "plugin.rst", "troubleshooting.rst", "tutorial.rst"], "titles": ["API", "Command Line Utilities", "Concepts", "Welcome to Earth2 MIP\u2019s documentation!", "Bring your own model", "Troubleshooting", "Tutorial"], "terms": {"while": [1, 2], "one": [1, 2, 4], "can": [1, 2, 4], "certainli": 1, "write": 1, "own": [1, 3], "script": [1, 4], "us": [1, 2, 4], "tool": [1, 2], "api": [1, 2, 3], "thi": [1, 2, 4, 5], "ha": [1, 2, 4], "overhead": 1, "For": [1, 2, 4], "each": [1, 2], "need": [1, 2, 4], "initi": [1, 2], "torch": [1, 2, 4], "distribut": 1, "group": 1, "run": [1, 2], "parallel": 1, "submiss": 1, "cluster": [1, 4], "To": [1, 2], "simplifi": 1, "deploy": 1, "we": [1, 2], "also": [1, 2], "includ": 1, "cli": [1, 4], "corresond": 1, "common": [1, 2], "python": [1, 2, 4], "m": [1, 4], "earth2mip": [1, 2, 4], "inference_medium_rang": [1, 4], "score_determinist": 1, "inference_ensembl": 1, "run_infer": 1, "lagged_ensembl": 1, "earth2": 2, "mip": [2, 4], "follow": 2, "layer": 2, "abstract": [2, 4], "wrap": [2, 4], "machin": 2, "learn": 2, "load": [2, 4], "condit": 2, "score": [2, 4], "against": 2, "work": [2, 4], "do": [2, 4], "thing": 2, "like": [2, 4], "ensembl": 2, "infer": [2, 4, 5], "allow": 2, "reproduc": 2, "checkpoint": 2, "from": [2, 4], "disk": 2, "cloud": 2, "command": [2, 3, 4], "line": [2, 3, 4], "job": 2, "paral": 2, "environ": [2, 4], "The": [2, 4], "core": 2, "relat": 2, "ar": [2, 4], "e": 2, "g": 2, "nn": [2, 4], "time_loop": [2, 4], "timeloop": [2, 4], "present": 2, "increasingli": 2, "minim": 2, "interfac": [2, 3], "demonstr": 2, "persist": [2, 4], "A": 2, "alwai": 2, "return": [2, 4], "It": [2, 4], "baselin": 2, "weather": [2, 4], "At": 2, "root": 2, "input": 2, "output": 2, "correspond": [2, 4], "two": [2, 4], "dimension": 2, "field": 2, "defin": 2, "planet": 2, "These": 2, "reifi": 2, "some": [2, 4], "arrai": 2, "structur": 2, "name": [2, 4, 5], "channel": [2, 4], "grid": [2, 4], "object": 2, "schema": 2, "fcn": [2, 4], "provid": [2, 4, 5], "metadata": [2, 4], "about": 2, "s": 2, "see": [2, 4, 5], "here": 2, "how": 2, "implement": [2, 4], "import": [2, 4], "class": [2, 4], "persistencemodul": 2, "def": [2, 4], "forward": [2, 4], "self": [2, 4], "x": [2, 4], "veri": 2, "simpl": 2, "an": [2, 4], "outsid": 2, "perspect": 2, "difficult": 2, "more": [2, 4], "semant": 2, "mean": 2, "requir": [2, 4], "higher": 2, "level": [2, 4], "encapsul": 2, "timestep": 2, "logic": [2, 4], "preprocess": 2, "network": [2, 4], "turn": 2, "datetim": 2, "peristencetimeloop": 2, "time_step": 2, "timedelta": 2, "hour": 2, "12": 2, "1": [2, 4], "histori": 2, "onli": [2, 4], "current": [2, 4], "n_history_level": 2, "in_channel_nam": 2, "b": 2, "c": 2, "out_channel_nam": 2, "grid_721x1440": 2, "__call__": 2, "restart": 2, "none": [2, 4], "h": 2, "w": 2, "shape": 2, "assert": [2, 4], "len": 2, "true": [2, 4], "yield": 2, "step": [2, 4], "expos": 2, "other": [2, 4], "doe": 2, "support": [2, 4], "capabl": 2, "mani": 2, "algorithm": 2, "most": 2, "easili": 2, "express": 2, "oper": 2, "over": 2, "2d": 2, "state": 2, "call": [2, 4], "row": 2, "column": 2, "lead": 2, "size": 2, "mai": 2, "unbound": 2, "exampl": [2, 4], "comput": 2, "depend": 2, "metric": 2, "rmse": 2, "averag": 2, "squar": 2, "differ": 2, "observ": 2, "dimens": 2, "compar": 2, "ani": 2, "handl": 2, "One": [2, 4], "advantag": 2, "archiv": 2, "repres": 2, "xarrayforecast": 2, "same": 2, "code": [2, 4], "both": 2, "static": 2, "stream": 2, "final": [2, 4], "begin": 2, "jan": 2, "2018": 2, "produc": 2, "ic": 2, "everi": 2, "sampl": 2, "peristenceforecast": 2, "channel_nam": 2, "__init__": 2, "initial_data": 2, "map": 2, "np": 2, "ndarrai": 2, "__getitem__": 2, "i": 2, "initial_tim": 2, "lead_dt": 2, "init_dt": 2, "creat": [2, 4], "around": 2, "normal": 2, "center": 2, "zero": 2, "3": 2, "scale": 2, "ones": 2, "note": 2, "n_histori": 2, "0": [2, 4], "you": [2, 3, 5], "timeloopforecast": 2, "721": 2, "1440": 2, "initial_condit": [2, 4], "era5": [2, 4], "hdf5datasourc": [2, 4], "directori": 2, "contain": [2, 4], "json": [2, 4], "file": [2, 4], "typic": 2, "paramet": 2, "constant": 2, "etc": [2, 4], "tutori": 3, "concept": [3, 4], "model": 3, "wrapper": 3, "forecast": [3, 4], "translat": 3, "between": 3, "data": 3, "sourc": [3, 4], "packag": [3, 4], "util": [3, 4], "bring": 3, "your": 3, "modul": 3, "time": 3, "loop": 3, "troubleshoot": 3, "index": 3, "search": 3, "page": 3, "option": 4, "first": 4, "build": 4, "function": 4, "todo": 4, "add": 4, "cross": 4, "refer": 4, "pangu_24": 4, "py": 4, "howev": 4, "sometim": 4, "conveni": 4, "especi": 4, "when": 4, "deploi": 4, "so": 4, "under": [4, 6], "hood": 4, "all": 4, "get_model": 4, "modifi": 4, "behavior": 4, "either": 4, "specifi": [4, 5], "entrypoint": 4, "which": 4, "archicture_entrypoint": 4, "method": 4, "gener": 4, "often": 4, "easier": 4, "frequent": 4, "train": 4, "base": 4, "alreadi": 4, "have": 4, "repurpos": 4, "lower": 4, "select": 4, "architecture_entrypoint": 4, "kei": 4, "plugin": 4, "would": 4, "my_packag": 4, "load_persistence_modul": 4, "pretrain": 4, "devic": 4, "arg": 4, "fcn_mip": 4, "model_registri": 4, "get": 4, "path": 4, "open": 4, "weight": 4, "Then": 4, "ab": 4, "should": 4, "channel_set": 4, "var73": 4, "721x1440": 4, "in_channel": 4, "2": 4, "out_channel": 4, "valu": 4, "thei": 4, "form": 4, "abl": 4, "Or": 4, "variabl": 4, "onc": 4, "pass": 4, "torchrun": 4, "nproc_per_nod": 4, "ngpu": 4, "n": 4, "56": 4, "nc": 4, "var34": 4, "set": 4, "project": 4, "loader": 4, "pangu": 4, "doesn": 4, "t": 4, "matter": 4, "sub": 4, "p6": 4, "pangu_weather_6": 4, "onnx": 4, "p24": 4, "pangu_weather_24": 4, "model_6": 4, "pangustack": 4, "panguweath": 4, "model_24": 4, "panguinfer": 4, "kwarg": 4, "time_step_hour": 4, "6": 4, "interleav": 4, "load_24substep6": 4, "custom": 4, "userwarn": 5, "cudaexecutionprovid": 5, "avail": 5, "cpuexecutionprovid": 5, "If": 5, "warn": 5, "fail": 5, "gpu": 5, "fix": 5, "onnxruntim": 5, "instal": 5, "pip": 5, "uninstal": 5, "construct": 6}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"api": 0, "command": 1, "line": 1, "util": 1, "concept": 2, "model": [2, 4], "wrapper": 2, "modul": [2, 4], "time": [2, 4], "loop": [2, 4], "forecast": 2, "translat": 2, "between": 2, "data": [2, 4], "sourc": 2, "packag": 2, "welcom": 3, "earth2": 3, "mip": 3, "s": 3, "document": 3, "content": 3, "indic": 3, "tabl": 3, "bring": 4, "your": 4, "own": 4, "interfac": 4, "you": 4, "troubleshoot": 5, "tutori": 6}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 56}})