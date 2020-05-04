# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sketch-RNN-Plus Data Utilities."""

# internal imports

import numpy as np
import matplotlib.pyplot as plt
#from scipy.misc import imresize as resize
from PIL import Image
from IPython.display import SVG, display
import svgwrite # conda install -c omnia svgwrite=1.1.6
import os
import json
import random

from numba import jit
from numpy import arange

# Plot helper functions

IMAGE_SIZE = 64

# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.045, svg_filename = '/tmp/sketch_rnn/svg/sample.svg'):
  if not os.path.exists(os.path.dirname(svg_filename)):
    os.makedirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  display(SVG(dwg.tostring()))


def show_image(img):
  plt.imshow(1-img.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
  plt.show()

# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=5.0, grid_space_x=5.0):
  def get_start_and_end(x):
    x = np.array(x)
    x = x[:, 0:2]
    x_start = x[0]
    x_end = x.sum(axis=0)
    x = x.cumsum(axis=0)
    x_max = x.max(axis=0)
    x_min = x.min(axis=0)
    center_loc = (x_max+x_min)*0.5
    return x_start-center_loc, x_end
  x_pos = 0.0
  y_pos = 0.0
  result = [[x_pos, y_pos, 1]]
  for sample in s_list:
    s = np.array([[0, 0, 1]] + sample[0].tolist())
    grid_loc = sample[1]
    grid_y = grid_loc[0]*grid_space+grid_space*0.5
    grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
    start_loc, delta_pos = get_start_and_end(s)

    loc_x = start_loc[0]
    loc_y = start_loc[1]
    new_x_pos = grid_x+loc_x
    new_y_pos = grid_y+loc_y
    result.append([new_x_pos-x_pos, new_y_pos-y_pos, 1])

    result += s.tolist()
    result[-1][2] = 1
    x_pos = new_x_pos+delta_pos[0]
    y_pos = new_y_pos+delta_pos[1]
  return np.array(result)

def read_categories(filename):
  with open(filename) as f:
    content = f.readlines()
  content = [x.strip() for x in content]
  return content

def get_config(config_dir):
  chosen_classes = read_categories(os.path.join(config_dir, 'category.txt'))
  with open(os.path.join(config_dir, 'config.json'), 'r') as fp:
    config = json.load(fp)
  config['numClasses'] = len(chosen_classes)
  config['classList'] = chosen_classes
  return config

def get_bounds(data, factor=1):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0]) / factor
    y = float(data[i, 1]) / factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)


def slerp(p0, p1, t):
  """Spherical interpolation."""
  omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
  so = np.sin(omega)
  return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
  """Linear interpolation."""
  return (1.0 - t) * p0 + t * p1


# A note on formats:
# Sketches are encoded as a sequence of strokes. stroke-3 and stroke-5 are
# different stroke encodings.
#   stroke-3 uses 3-tuples, consisting of x-offset, y-offset, and a binary
#       variable which is 1 if the pen is lifted between this position and
#       the next, and 0 otherwise.
#   stroke-5 consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
#   one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
#   See section 3.1 of https://arxiv.org/abs/1704.03477 for more detail.
# Sketch-RNN takes input in stroke-5 format, with sketches padded to a common
# maximum length and prefixed by the special start token [0, 0, 1, 0, 0]
# The QuickDraw dataset is stored using stroke-3.
def strokes_to_lines(strokes):
  """Convert stroke-3 format to polyline format."""
  x = 0
  y = 0
  lines = []
  line = []
  for i in range(len(strokes)):
    if strokes[i, 2] == 1:
      x += float(strokes[i, 0])
      y += float(strokes[i, 1])
      line.append([x, y])
      lines.append(line)
      line = []
    else:
      x += float(strokes[i, 0])
      y += float(strokes[i, 1])
      line.append([x, y])
  return lines


def raw_to_lines(raw):
  """Convert raw QuickDraw format to polyline format."""
  result = []
  N = len(raw)
  for i in range(N):
    line = []
    rawline = raw[i]
    M = len(rawline[0])
    for j in range(M):
      line.append([rawline[0][j], rawline[1][j]])
    result.append(line)
  return result


def lines_to_strokes(lines):
  """Convert polyline format to stroke-3 format."""
  eos = 0
  strokes = [[0, 0, 0]]
  for line in lines:
    linelen = len(line)
    for i in range(linelen):
      eos = 0 if i < linelen - 1 else 1
      strokes.append([line[i][0], line[i][1], eos])
  strokes = np.array(strokes)
  strokes[1:, 0:2] -= strokes[:-1, 0:2]
  return strokes[1:, :]


def augment_strokes(strokes, prob=0.0):
  """Perform data augmentation by randomly dropping out strokes."""
  # drop each point within a line segments with a probability of prob
  # note that the logic in the loop prevents points at the ends to be dropped.
  result = []
  prev_stroke = [0, 0, 1]
  count = 0
  stroke = [0, 0, 1]  # Added to be safe.
  for i in range(len(strokes)):
    candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
    if candidate[2] == 1 or prev_stroke[2] == 1:
      count = 0
    else:
      count += 1
    urnd = np.random.rand()  # uniform random variable
    if candidate[2] == 0 and prev_stroke[2] == 0 and count > 3 and urnd < prob:
      stroke[0] += candidate[0]
      stroke[1] += candidate[1]
    else:
      stroke = candidate
      prev_stroke = stroke
      result.append(stroke)
  return np.array(result)

def random_scale_strokes(data, random_scale_factor=0.0):
  """Augment data by stretching x and y axis randomly [1-2*e, 1]."""
  x_scale_factor = (
      np.random.random() - 0.5) * 2 * random_scale_factor + 1.0 - random_scale_factor
  y_scale_factor = (
      np.random.random() - 0.5) * 2 * random_scale_factor + 1.0 - random_scale_factor
  result = np.copy(data)
  result[:, 0] *= x_scale_factor
  result[:, 1] *= y_scale_factor
  return result

# from https://stackoverflow.com/questions/44159861/how-do-i-parse-this-ndjson-file-in-python
@jit
def get_line(x1, y1, x2, y2):
  points = []
  issteep = abs(y2-y1) > abs(x2-x1)
  if issteep:
    x1, y1 = y1, x1
    x2, y2 = y2, x2
  rev = False
  if x1 > x2:
    x1, x2 = x2, x1
    y1, y2 = y2, y1
    rev = True
  deltax = x2 - x1
  deltay = abs(y2-y1)
  error = int(deltax / 2)
  y = y1
  ystep = None
  if y1 < y2:
    ystep = 1
  else:
    ystep = -1
  for x in arange(x1, x2 + 1):
    if issteep:
      points.append((y, x))
    else:
      points.append((x, y))
    error -= deltay
    if error < 0:
      y += ystep
      error += deltax
  # Reverse the list if the coordinates were reversed
  if rev:
    points.reverse()
  return points   

@jit
def stroke_to_quickdraw(orig_data, max_dim_size=5.0):
  ''' convert back to list of points format, up to 255 dimensions '''
  data = np.copy(orig_data)
  data[:, 0:2] *= (255.0/max_dim_size) # to prevent overflow
  data = np.round(data).astype(np.int)
  line = []
  lines = []
  abs_x = 0
  abs_y = 0
  for i in arange(0, len(data)):
    dx = data[i,0]
    dy = data[i,1]
    abs_x += dx
    abs_y += dy
    abs_x = np.maximum(abs_x, 0)
    abs_x = np.minimum(abs_x, 255)
    abs_y = np.maximum(abs_y, 0)
    abs_y = np.minimum(abs_y, 255)  
    lift_pen = data[i, 2]
    line.append([abs_x, abs_y])
    if (lift_pen == 1):
      lines.append(line)
      line = []
  return lines

@jit
def create_image(stroke3, max_dim_size=5.0):
  image_dim = IMAGE_SIZE
  factor = 256/image_dim
  pixels = np.zeros((image_dim, image_dim))
  
  sketch = stroke_to_quickdraw(stroke3, max_dim_size=max_dim_size)

  x = -1
  y = -1

  for stroke in sketch:
    for i in arange(len(stroke)):
      if x != -1: 
        for point in get_line(stroke[i][0], stroke[i][1], x, y):
          pixels[int(point[0]/factor),int(point[1]/factor)] = 1
      pixels[int(stroke[i][0]/factor),int(stroke[i][1]/factor)] = 1
      x = stroke[i][0]
      y = stroke[i][1]
    x = -1
    y = -1
  return pixels.T.reshape(image_dim, image_dim, 1)

def package_augmentation(strokes,
                         random_drop_factor=0.15,
                         random_scale_factor=0.15,
                         max_dim_size=5.0):
  test_stroke = random_scale_strokes(
    augment_strokes(strokes, random_drop_factor),
    random_scale_factor)
  min_x, max_x, min_y, max_y = get_bounds(test_stroke, factor=1)
  rand_offset_x = (max_dim_size-max_x+min_x)*np.random.rand()
  rand_offset_y = (max_dim_size-max_y+min_y)*np.random.rand()
  test_stroke[0][0] += rand_offset_x
  test_stroke[0][1] += rand_offset_y
  return test_stroke

def scale_bound(stroke, average_dimension=10.0):
  """Scale an entire image to be less than a certain size."""
  # stroke is a numpy array of [dx, dy, pstate], average_dimension is a float.
  # modifies stroke directly.
  bounds = get_bounds(stroke, 1)
  max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
  stroke[:, 0:2] /= (max_dimension / average_dimension)


def to_normal_strokes(big_stroke):
  """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
  l = 0
  for i in range(len(big_stroke)):
    if big_stroke[i, 4] > 0:
      l = i
      break
  if l == 0:
    l = len(big_stroke)
  result = np.zeros((l, 3))
  result[:, 0:2] = big_stroke[0:l, 0:2]
  result[:, 2] = big_stroke[0:l, 3]
  return result


def clean_strokes(sample_strokes, factor=100):
  """Cut irrelevant end points, scale to pixel space and store as integer."""
  # Useful function for exporting data to .json format.
  copy_stroke = []
  added_final = False
  for j in range(len(sample_strokes)):
    finish_flag = int(sample_strokes[j][4])
    if finish_flag == 0:
      copy_stroke.append([
          int(round(sample_strokes[j][0] * factor)),
          int(round(sample_strokes[j][1] * factor)),
          int(sample_strokes[j][2]),
          int(sample_strokes[j][3]), finish_flag
      ])
    else:
      copy_stroke.append([0, 0, 0, 0, 1])
      added_final = True
      break
  if not added_final:
    copy_stroke.append([0, 0, 0, 0, 1])
  return copy_stroke


def to_big_strokes(stroke, max_len=250):
  """Converts from stroke-3 to stroke-5 format and pads to given length."""
  # (But does not insert special start token).

  result = np.zeros((max_len, 5), dtype=float)
  l = len(stroke)
  assert l <= max_len
  result[0:l, 0:2] = stroke[:, 0:2]
  result[0:l, 3] = stroke[:, 2]
  result[0:l, 2] = 1 - result[0:l, 3]
  result[l:, 4] = 1
  return result


def get_max_len(strokes):
  """Return the maximum length of an array of strokes."""
  max_len = 0
  for stroke in strokes:
    ml = len(stroke)
    if ml > max_len:
      max_len = ml
  return max_len


def get_min_len(strokes):
  """Return the minimum length of an array of strokes."""
  min_len = 1000000000
  for stroke in strokes:
    ml = len(stroke)
    if ml < min_len:
      min_len = ml
  return min_len


class DataLoader(object):
  """Class for loading data."""

  def __init__(self,
               stroke_sets,
               process_images=True,
               batch_size=100,
               max_seq_length=100,
               scale_factor=0.1,
               random_scale_factor=0.0,
               augment_stroke_prob=0.0,
               limit=1000):
    # process stroke_sets:
    self.set_len = []
    for s in stroke_sets:
      self.set_len.append(len(s))
    strokes = np.concatenate(stroke_sets, axis=0)
    self.num_sets = len(self.set_len)
    self.set_ranges = np.concatenate([[0],np.cumsum(self.set_len)])
    self.batch_size = batch_size  # minibatch size
    self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
    self.max_dim_size = 1.0/scale_factor # maximum dimension of stroke image
    self.scale_factor = scale_factor  # divide offsets by this factor
    self.random_scale_factor = random_scale_factor  # data augmentation method
    # Removes large gaps in the data. x and y offsets are clamped to have
    # absolute value no greater than this limit.
    self.limit = limit
    self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
    self.start_stroke_token = [0, 0, 0, 1, 0]  # S_0 in sketch-rnn paper
    # sets self.strokes (list of ndarrays, one per sketch, in stroke-3 format,
    # sorted by size)
    self.process_images = process_images
    self.preprocess(strokes)

  def preprocess(self, strokes):
    """Remove entries from strokes having > max_seq_length points."""
    raw_data = []
    count_data = 0

    for i in range(len(strokes)):
      data = strokes[i]
      if len(data) <= (self.max_seq_length):
        count_data += 1
        # removes large gaps from the data
        data = np.minimum(data, self.limit)
        data = np.maximum(data, -self.limit)
        data = np.array(data, dtype=np.float32)
        data[:, 0:2] /= self.scale_factor
        raw_data.append(data)
      else:
        assert False, "error: datapoint length >"+str(self.max_seq_length)
    self.strokes = raw_data
    print("total drawings <= max_seq_len is %d" % count_data)
    self.num_batches = int(count_data / self.batch_size)

  def random_sample(self):
    """Return a random sample, in stroke-3 format as used by draw_strokes."""
    sample = np.copy(random.choice(self.strokes))
    return sample

  def calculate_normalizing_scale_factor(self):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(self.strokes)):
      if len(self.strokes[i]) > self.max_seq_length:
        continue
      for j in range(len(self.strokes[i])):
        data.append(self.strokes[i][j, 0])
        data.append(self.strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

  def normalize(self, scale_factor=None):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    if scale_factor is None:
      scale_factor = self.calculate_normalizing_scale_factor()
    self.scale_factor = scale_factor
    for i in range(len(self.strokes)):
      self.strokes[i][:, 0:2] /= self.scale_factor

  def _get_batch_from_indices(self, indices):
    """Given a list of indices, return the potentially augmented batch."""
    x_batch = []
    image_batch = []
    seq_len = []
    for idx in range(len(indices)):
      i = indices[idx]
      data_copy = package_augmentation(self.strokes[i],
                                       random_drop_factor=self.augment_stroke_prob,
                                       random_scale_factor=self.random_scale_factor,
                                       max_dim_size=self.max_dim_size)
      if self.process_images:
        image_batch.append(create_image(data_copy, max_dim_size=self.max_dim_size))
      x_batch.append(data_copy)
      length = len(data_copy)
      seq_len.append(length)
    seq_len = np.array(seq_len, dtype=int)
    # We return four things: stroke-3 format, stroke-5 format, list of seq_len, pixel_images
    return [x_batch, self.pad_batch(x_batch, self.max_seq_length), seq_len, image_batch]

  def random_batch(self):
    """Return a randomised portion of the training data."""
    # naive method just randomizes indices, i.e.:
    # idx = np.random.choice(np.arange(len(self.strokes)), self.batch_size)
    # to avoid unbalanced datasets, we randomize on categories first:
    batch_cat = np.random.randint(self.num_sets, size=self.batch_size)
    idx = []
    for i in range(self.batch_size):
      category = batch_cat[i]
      idx_lo = self.set_ranges[category]
      idx_hi = self.set_ranges[category+1]
      idx.append(np.random.randint(idx_lo, idx_hi))
    return self._get_batch_from_indices(idx) + [batch_cat]

  def get_batch(self, idx):
    """Get the idx'th batch from the dataset."""
    assert idx >= 0, "idx must be non negative"
    assert idx < self.num_batches, "idx must be less than the number of batches"
    start_idx = idx * self.batch_size
    indices = range(start_idx, start_idx + self.batch_size)
    return self._get_batch_from_indices(indices)

  def pad_batch(self, batch, max_len):
    """Pad the batch to be stroke-5 bigger format as described in paper."""
    result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
    assert len(batch) == self.batch_size
    for i in range(self.batch_size):
      l = len(batch[i])
      assert l <= max_len
      result[i, 0:l, 0:2] = batch[i][:, 0:2]
      result[i, 0:l, 3] = batch[i][:, 2]
      result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
      result[i, l:, 4] = 1
      # put in the first token, as described in sketch-rnn methodology
      result[i, 1:, :] = result[i, :-1, :]
      result[i, 0, :] = 0
      result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
      result[i, 0, 3] = self.start_stroke_token[3]
      result[i, 0, 4] = self.start_stroke_token[4]
    return result

def process_dataset(data_set, max_len, min_len):
  fixed_data = []
  all_length = []
  all_size = []
  for data in data_set:
    len_data = len(data)
    if len_data >= max_len or len_data < min_len:
      continue
    min_x, max_x, min_y, max_y = get_bounds(data)
    all_length.append(len(data))
    t = np.concatenate([[[-min_x, -min_y, 0]], data], axis=0).astype(np.float)
    factor = np.max([max_x-min_x, max_y-min_y])
    t[:, 0:2] /= factor
    fixed_data.append(t)
  return fixed_data

def get_dataset(class_name, max_len, min_len):
  print('loading', class_name)
  filename = os.path.join('npz', class_name+'.full.npz')
  load_data = np.load(filename, encoding='latin1')
  train_set_data = load_data['train']
  valid_set_data = load_data['valid']
  test_set_data = load_data['test']

  train_set_data = process_dataset(train_set_data, max_len, min_len)
  valid_set_data = process_dataset(valid_set_data, max_len, min_len)
  test_set_data = process_dataset(test_set_data, max_len, min_len)

  return train_set_data, valid_set_data, test_set_data

def get_test_dataset(class_name, max_len, min_len):
  print('loading', class_name)
  filename = os.path.join('npz', class_name+'.full.npz')
  load_data = np.load(filename, encoding='latin1')
  test_set_data = load_data['test']
  test_set_data = process_dataset(test_set_data, max_len, min_len)
  return test_set_data

def get_dataset_list(class_list, max_len, min_len):
  train = []
  valid = []
  test = []
  for c in class_list:
    train_set_data, valid_set_data, test_set_data = get_dataset(c, max_len, min_len)
    print("count (train/valid/test):",
          len(train_set_data),
          len(valid_set_data),
          len(test_set_data))
    train.append(train_set_data)
    valid.append(valid_set_data)
    test.append(test_set_data)
  return train, valid, test

def get_test_dataset_list(class_list, max_len, min_len):
  test = []
  for c in class_list:
    test_set_data = get_test_dataset(c, max_len, min_len)
    print("count (test):",
          len(test_set_data))
    test.append(test_set_data)
  return test
