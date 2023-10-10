# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

channel_means = {
    "u10m": -0.06854962557554245,
    "v10m": 0.18743863701820374,
    "u100m": -0.009113931097090244,
    "v100m": 0.19518691301345825,
    "t2m": 278.7031555175781,
    "sp": 96653.515625,
    "msl": 100957.6796875,
    "tcwv": 18.46604347229004,
    "u1": 10.748488426208496,
    "u2": 9.040291786193848,
    "u3": 7.640656471252441,
    "u5": 6.190047264099121,
    "u7": 5.613990306854248,
    "u10": 4.81683874130249,
    "u20": 3.7416577339172363,
    "u30": 3.9482882022857666,
    "u50": 5.662464618682861,
    "u70": 7.420891761779785,
    "u100": 10.338683128356934,
    "u125": 12.378543853759766,
    "u150": 13.560766220092773,
    "u175": 14.09630012512207,
    "u200": 14.167032241821289,
    "u225": 13.864096641540527,
    "u250": 13.288179397583008,
    "u300": 11.738770484924316,
    "u350": 10.155152320861816,
    "u400": 8.77362060546875,
    "u450": 7.578890800476074,
    "u500": 6.538493633270264,
    "u550": 5.618127822875977,
    "u600": 4.793304920196533,
    "u650": 4.032490253448486,
    "u700": 3.3098866939544678,
    "u750": 2.626377820968628,
    "u775": 2.299865484237671,
    "u800": 1.9830362796783447,
    "u825": 1.6734198331832886,
    "u850": 1.3726757764816284,
    "u875": 1.0908571481704712,
    "u900": 0.8307265639305115,
    "u925": 0.5900797843933105,
    "u950": 0.37076279520988464,
    "u975": 0.17277194559574127,
    "u1000": -0.05516379326581955,
    "v1": -0.081306092441082,
    "v2": -0.021461497992277145,
    "v3": -0.020173462107777596,
    "v5": -0.00731061352416873,
    "v7": -0.004137726966291666,
    "v10": -0.0019085388630628586,
    "v20": -0.00026605380116961896,
    "v30": 0.0010730322683230042,
    "v50": 0.006631731055676937,
    "v70": 0.0045732553116977215,
    "v100": 0.0040822383016347885,
    "v125": -0.02326071448624134,
    "v150": -0.045998428016901016,
    "v175": -0.0552876852452755,
    "v200": -0.03847499191761017,
    "v225": -0.02821849100291729,
    "v250": -0.027320099994540215,
    "v300": -0.02411925420165062,
    "v350": -0.01592441089451313,
    "v400": -0.019759656861424446,
    "v450": -0.026930494233965874,
    "v500": -0.02907356433570385,
    "v550": -0.031823400408029556,
    "v600": -0.02572263590991497,
    "v650": -0.009626002050936222,
    "v700": 0.02697981894016266,
    "v750": 0.06845545023679733,
    "v775": 0.0888051986694336,
    "v800": 0.10753155499696732,
    "v825": 0.12397794425487518,
    "v850": 0.14166374504566193,
    "v875": 0.1662677526473999,
    "v900": 0.19087384641170502,
    "v925": 0.20176121592521667,
    "v950": 0.19572316110134125,
    "v975": 0.18371663987636566,
    "v1000": 0.18469774723052979,
    "w1": -1.1315158872093889e-06,
    "w2": -8.018057087610941e-06,
    "w3": -1.537994648970198e-05,
    "w5": -2.478267197147943e-05,
    "w7": -3.223319072276354e-05,
    "w10": -4.135979543207213e-05,
    "w20": -5.539332050830126e-05,
    "w30": -5.9556386986514553e-05,
    "w50": -5.383044117479585e-05,
    "w70": -4.982736209058203e-05,
    "w100": -2.3971153495949693e-05,
    "w125": 3.87470163332182e-06,
    "w150": 1.66643312695669e-05,
    "w175": 1.5288433132809587e-05,
    "w200": 9.045768820215017e-06,
    "w225": 6.168767413328169e-06,
    "w250": 1.4241597455111332e-05,
    "w300": 9.39343444770202e-05,
    "w350": 0.00019888172391802073,
    "w400": 0.000272307894192636,
    "w450": 0.0002964134037028998,
    "w500": 0.00030192555277608335,
    "w550": 0.0002988489577546716,
    "w600": 0.0003334358334541321,
    "w650": 0.0006918106810189784,
    "w700": 0.002709059976041317,
    "w750": 0.005279306787997484,
    "w775": 0.006850168574601412,
    "w800": 0.008400746621191502,
    "w825": 0.010028586722910404,
    "w850": 0.011793135665357113,
    "w875": 0.01368639525026083,
    "w900": 0.015604125335812569,
    "w925": 0.01744643785059452,
    "w950": 0.019302070140838623,
    "w975": 0.020728887990117073,
    "w1000": 0.020804615691304207,
    "z1": 463206.4375,
    "z2": 411553.125,
    "z3": 382229.46875,
    "z5": 346674.875,
    "z7": 324015.15625,
    "z10": 300542.71875,
    "z20": 256143.234375,
    "z30": 230781.734375,
    "z50": 199379.265625,
    "z70": 179031.015625,
    "z100": 157709.578125,
    "z125": 144308.1875,
    "z150": 133217.546875,
    "z175": 123719.1875,
    "z200": 115399.328125,
    "z225": 107985.0703125,
    "z250": 101282.53125,
    "z300": 89464.4921875,
    "z350": 79181.1875,
    "z400": 70015.5625,
    "z450": 61723.00390625,
    "z500": 54140.7421875,
    "z550": 47151.71484375,
    "z600": 40667.58984375,
    "z650": 34617.66796875,
    "z700": 28943.169921875,
    "z750": 23597.029296875,
    "z775": 21036.0859375,
    "z800": 18544.6875,
    "z825": 16119.4697265625,
    "z850": 13757.439453125,
    "z875": 11455.5458984375,
    "z900": 9210.625,
    "z925": 7019.46533203125,
    "z950": 4878.94189453125,
    "z975": 2786.269775390625,
    "z1000": 738.6001586914062,
    "t1": 261.1324462890625,
    "t2": 255.74183654785156,
    "t3": 247.80210876464844,
    "t5": 237.51869201660156,
    "t7": 231.79122924804688,
    "t10": 226.91737365722656,
    "t20": 219.69107055664062,
    "t30": 216.14366149902344,
    "t50": 212.13992309570312,
    "t70": 209.16539001464844,
    "t100": 208.21510314941406,
    "t125": 210.52633666992188,
    "t150": 213.35560607910156,
    "t175": 215.93296813964844,
    "t200": 218.19093322753906,
    "t225": 220.4385986328125,
    "t250": 222.94039916992188,
    "t300": 229.04507446289062,
    "t350": 235.85696411132812,
    "t400": 242.3114013671875,
    "t450": 248.05335998535156,
    "t500": 253.07688903808594,
    "t550": 257.4590759277344,
    "t600": 261.2414855957031,
    "t650": 264.5358581542969,
    "t700": 267.5036926269531,
    "t750": 270.2417297363281,
    "t775": 271.4933776855469,
    "t800": 272.6558532714844,
    "t825": 273.7077331542969,
    "t850": 274.6662902832031,
    "t875": 275.5856018066406,
    "t900": 276.5234680175781,
    "t925": 277.5076599121094,
    "t950": 278.5635681152344,
    "t975": 279.78955078125,
    "t1000": 281.2162170410156,
    "r1": 0.0012648305855691433,
    "r2": 0.006784914992749691,
    "r3": 0.021754322573542595,
    "r5": 0.10752345621585846,
    "r7": 0.2999992072582245,
    "r10": 0.8303617835044861,
    "r20": 3.4991626739501953,
    "r30": 4.6645050048828125,
    "r50": 6.580825328826904,
    "r70": 13.205595970153809,
    "r100": 26.424592971801758,
    "r125": 26.096942901611328,
    "r150": 26.452449798583984,
    "r175": 29.685396194458008,
    "r200": 35.18281936645508,
    "r225": 41.35318374633789,
    "r250": 46.84218978881836,
    "r300": 53.30179214477539,
    "r350": 53.954071044921875,
    "r400": 52.459781646728516,
    "r450": 51.048095703125,
    "r500": 50.507022857666016,
    "r550": 50.88429260253906,
    "r600": 51.785911560058594,
    "r650": 53.15390396118164,
    "r700": 55.32671356201172,
    "r750": 58.33307647705078,
    "r775": 60.228816986083984,
    "r800": 62.532310485839844,
    "r825": 65.50992584228516,
    "r850": 69.1156234741211,
    "r875": 72.80475616455078,
    "r900": 76.10183715820312,
    "r925": 78.84436798095703,
    "r950": 80.54322052001953,
    "r975": 79.97803497314453,
    "r1000": 77.97394561767578,
    "2d": 274.1190185546875,
    "sst": 189.70262145996094,
    "skt": 279.1072692871094,
    "stl1": 279.7412414550781,
    "swvl1": 0.08473234623670578,
}
