# Copyright (c) 2017 Jes Frellsen and Wouter Boomsma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import collections
import numpy as np
from numpy.lib.arraypad import _validate_lengths
import tensorflow as tf

__all__ = ['tf_pad_wrap']

_DummyArray = collections.namedtuple("_DummyArray", ["ndim"])


def _pad_wrap(arr, pad_amt, axis=-1):
    """
    Modified from numpy.lib.arraypad._pad_wrap
    """

    # Implicit booleanness to test for zero (or None) in any scalar type
    if pad_amt[0] == 0 and pad_amt[1] == 0:
        return arr

    ##########################################################################
    # Prepended region

    # Slice off a reverse indexed chunk from near edge to pad `arr` before
    start = arr.shape[axis] - pad_amt[0]
    end = arr.shape[axis]
    wrap_slice = tuple(slice(None) if i != axis else slice(start, end)
                       for (i, x) in enumerate(arr.shape))
    wrap_chunk1 = arr[wrap_slice]

    ##########################################################################
    # Appended region

    # Slice off a reverse indexed chunk from far edge to pad `arr` after
    wrap_slice = tuple(slice(None) if i != axis else slice(0, pad_amt[1])
                       for (i, x) in enumerate(arr.shape))
    wrap_chunk2 = arr[wrap_slice]

    # Concatenate `arr` with both chunks, extending along `axis`
    return tf.concat((wrap_chunk1, arr, wrap_chunk2), axis=axis)


def tf_pad_wrap(array, pad_width):
    """
    Modified from numpy.lib.arraypad.wrap
    """

    if not np.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')

    pad_width = _validate_lengths(_DummyArray(array.get_shape().ndims), pad_width)

    for axis, (pad_before, pad_after) in enumerate(pad_width):
        if array.get_shape().as_list()[axis] is None and (pad_before > 0 or pad_after > 0):
            raise TypeError('`pad_width` must be zero for dimensions that are None.')

    # If we get here, use new padding method
    newmat = tf.identity(array)

    for axis, (pad_before, pad_after) in enumerate(pad_width):
        # Recursive padding along any axis where `pad_amt` is too large
        # for indexing tricks. We can only safely pad the original axis
        # length, to keep the period of the reflections consistent.
        safe_pad = newmat.get_shape().as_list()[axis]

        if safe_pad is None:
            continue

        while ((pad_before > safe_pad) or
               (pad_after > safe_pad)):
            pad_iter_b = min(safe_pad,
                             safe_pad * (pad_before // safe_pad))
            pad_iter_a = min(safe_pad, safe_pad * (pad_after // safe_pad))
            newmat = _pad_wrap(newmat, (pad_iter_b, pad_iter_a), axis)

            pad_before -= pad_iter_b
            pad_after -= pad_iter_a
            safe_pad += pad_iter_b + pad_iter_a
        newmat = _pad_wrap(newmat, (pad_before, pad_after), axis)

    return newmat
