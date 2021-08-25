# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking


class AugmentedSimulator(object):
    """
    This class is the Base class that is used to create some "augmented simulator". These "augmented simulator" can be
    anything that emulates the behaviour of some "simulator".

    They are meant to use data coming from a `DataSet`
    """



