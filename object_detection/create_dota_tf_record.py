import tensorflow as tf
import object_detection.utils.utils as util
import sys
import os
import io
import PIL.Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

SAMPLES_PER_RECORD = 200


flags = tf.app.flags
flags.DEFINE_string('data_dir', r'pathtodota', 'Root directory to raw bod dataset.')
flags.DEFINE_string('label_map_path', r'',
                    'Path to label map proto')
FLAGS = flags.FLAGS
def create_tf_example(data,
                      imagepath,
                      label_map_dict,
                      filename,
                      ignore_difficult_instances=True
                      ):
  # TODO(user): Populate the following variables from your example.

  full_path = os.path.join(imagepath, filename + '.jpg')
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')

  #width = 1024
  #height = 1024
#   width = 608
#   height = 608
  width, height = image.size
  image_format = None # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  difficult_obj = []
  for obj in data:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue
    # if ((float(obj['bndbox'][0]) < 0) or
    #     (float(obj['bndbox'][1]) < 0) or
    #     (float(obj['bndbox'][2]) >= 1024) or
    #     (float(obj['bndbox'][3]) >= 1024) ):
    #     continue
    xmin = max(obj['bndbox'][0], 0)
    ymin = max(obj['bndbox'][1], 0)
    xmax = min(obj['bndbox'][2], width - 1)
    ymax = min(obj['bndbox'][3], height - 1)

    difficult_obj.append(int(difficult))

    xmins.append(float(xmin) / width)
    ymins.append(float(ymin) / height)
    xmaxs.append(float(xmax) / width)
    ymaxs.append(float(ymax) / height)

    # xmins.append(float(obj['bndbox'][0]) / width)
    # ymins.append(float(obj['bndbox'][1]) / height)
    # xmaxs.append(float(obj['bndbox'][2]) / width)
    # ymaxs.append(float(obj['bndbox'][3]) / height)

    classes_text.append(obj['name'].encode('utf8'))
    if (obj['name'] in label_map_dict):
        classes.append(label_map_dict[obj['name']])

    else:
        #print '>>>>>>>>>>>>>'
        continue


  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  #print 'tf_example: ', tf_example
  return tf_example


def get_output_filename(tf_records_name, fidx, file_num):
    return '%s.record-%05d-%05d' % (tf_records_name, fidx, file_num)


def tf_write(testortrain, tf_records_name):
    """
    :param testortrain: This is the index file for training data and test data. Please put them under the FLAGS.Data_dir path.
    :param tf_records_name:
    :return:
    """
    #writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'tf_records', tf_records_name))

    print ('start-------')
    # TODO(user): Write code to read in your dataset to examples variable
    data_dir = FLAGS.data_dir

    setname = os.path.join(data_dir, testortrain)
    imagepath = os.path.join(data_dir, 'JPEGImages')
    f = open(setname, 'r')
    lines = f.readlines()
    txtlist = [x.strip().replace(r'JPEGImages', r'wordlabel').replace('.jpg', '.txt') for x in lines]
    #txtlist = util.GetFileFromThisRootDir(os.path.join(data_dir, 'wordlabel'))
    i = 0
    fidx = 0
    file_num = len(txtlist)/SAMPLES_PER_RECORD + 1
    # for fullname, i in enumerate(txtlist):
    #     data = util.parse_bod_rec(fullname)
    #     #print 'len(data):', len(data)
    #     #print 'data:', data
    #     #assert len(data) >= 0, "there exists empty data: " + fullname
    #     basename = os.path.basename(os.path.splitext(fullname)[0])
    #     label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    #     #print 'label_map_dict', label_map_dict
    #     tf_example = create_tf_example(data,
    #                                    imagepath,
    #                                    label_map_dict,
    #                                    basename)
    #
    #     writer.write(tf_example.SerializeToString())
    # writer.close()
    while i < len(txtlist):
        # Open new TFRecord file.
        tf_filename = get_output_filename(tf_records_name, fidx, file_num)
        with tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'tf_records', tf_filename)) as tfrecord_writer:
            j = 0
            while i < len(txtlist) and j < SAMPLES_PER_RECORD:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(txtlist)))
                sys.stdout.flush()

                fullname = txtlist[i]
                data = util.parse_bod_rec(fullname)
                basename = os.path.basename(os.path.splitext(fullname)[0])
                label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
                tf_example = create_tf_example(data,
                                               imagepath,
                                               label_map_dict,
                                               basename)
                tfrecord_writer.write(tf_example.SerializeToString())
                i += 1
                j += 1
            fidx += 1
    print('\nFinished converting the DOTA dataset!')

def main(_):
    tf_write('val.txt', 'dota_val')
    tf_write('train.txt', 'dota_train')
    #tf_write('test.txt', 'dota_test_608.record')
    #tf_write('train.txt', 'dota_train_608.record')

if __name__ == '__main__':
  tf.app.run()
