from __future__ import division
import numpy as np
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer

from bounding_box_utils.bounding_box_utils import convert_coordinates

class AnchorBoxes(Layer):
    '''
    Tác dụng: Tạo ra một output tensor chứa tọa độ của các anchor box và các biến thể dựa trên input tensor.
    Một tập hợp các 2D anchor boxes được tạo ra dựa trên aspect ratios và scale trên mỗi một cells của grid cells. Các hộp được tham số hóa bằng các tọa độ `(xmin, xmax, ymin, ymax)`
    
    Input shape:
    - 4D tensor với shape `(batch, channels, height, width)`, nếu `dim_ordering = 'th'`
    hoặc `(batch, height, width, channels) nếu  `dim_ordering = 'tf'`.

    Output shape:
    - 5D tensor of shape `(batch, height, width, n_boxes, 8)`.
    chiều cuối cùng gồm 4 tọa độ của anchor box `(xmin, xmax, ymin, ymax)` và 4 tọa độ của biến thể `(xmin_variation, xmax_variation, ymin_variation, ymax_variation)`.
    '''
    def __init__( self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios = [0.5, 1.0, 2.0],
                 two_boxes_for_ar1 = True,
                 this_steps=None,
                 this_offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 coords='centroids',
                 normalize_coords=False,
                 **kwargs):
        '''
        Arguments:
        - `img_height`(int): Chiều cao của ảnh đầu vào.
        - `img_width`(int): Chiều rộng của ảnh đầu vào.
        - `this_scale`(float): Tỷ lệ của anchor box đầu tiên, một giá trị float thuộc [0,1], nhân tố scaling kích thước để tạo các anchor boxes dựa trên một tỷ lệ so với cạnh ngắn hơn trong width và height.
        - `next_scale`(float): giá trị tiếp theo cảu scale. được thiết lập khi và chỉ khi `self.two_boxes_for_ar1 == True`
        - `aspect_ratios`(list, optional): tập hợp các aspect ratios của các default boxes được tạo ra từ layer này.
        - `two_boxes_for_ar1`(bool, optional): được sử dụng chỉ khi aspect ratio = 1. 
                Nếu `True`, hai default boxes được tạo ra khi aspect ratio = 1. default box đầu tiên sử dụng scaling factor của layer tương ứng,
                default box thứ 2 sử dụng trung bình hình học giữa scaling factor và next scling factor.
        - `clip_boxes` (bool, optional: nếu `True`, các anchor boxes sẽ được cắt để nằm trong ảnh đầu vào.
        - `variance` (list, optional): tập hợp gồm 4 giá trị floats >0. là các anchor box offset tương ứng với mỗi tọa độ chia cho giá trị variances tương ứng của nó.
        - `coords` (str, optional): tọa độ của anchor box được sử dụng. Có thể là 'centroids' hoặc 'corners'. Mặc định là 'centroids'.
        - `normalize_coords` (bool, optional): nếu `True`, tọa độ của anchor box sẽ được chuẩn hóa về khoảng [0,1] dựa trên kích thước của ảnh đầu vào, tọa độ tương đối thay vì tuyệt đối.
        '''    
        if K.backend() != 'tensorflow':
            raise TypeError('This layer only supports TensorFlow at the moment, but you are using the {} backend.'.format(K.backend()))

        if( this_scale < 0) or (next_scale < 0) or (this_scale > 1): 
            raise ValueError('`this_scale` musst be in [0,1] and `next_scale` must be > 0, but `this_scale` == {}, `next_scale` == {}.'.format(this_scale, next_scale))

        if len(variances)!= 4:
            raise ValueError('4 variance values musst be passed, but {} values were received.'.format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError('All variance values must be > 0, but the following values were received: {}.'.format(variances))
        
        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords
        # Tính toán số lượng boxes trên 1 cell. TH aspect ratios = 1 thì thêm 1 box.
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        super(AnchorBoxes, self).build(input_shape)
    
    def call(self, x, mask=None):
        '''
        Return: Trả về anchor box tensor dựa trên shape của input tensor.
        
        Tensor này được thiết kế như là hằng số và không tham gia vào quá trình tính toán.
        
        Arguments:
            x(tensor): 4D tensor có shape `(batch, channels, height, width)` nếu `dim_ordering = 'th'` 
                hoặc `(batch, height, width, channels)` nếu `dim_ordering = 'tf'`.
                Input cho layer này phải là output của các localization predictor layer.

        '''
        # bước 1: tính toán width và height của box với mỗi aspect ratio
        ## cạnh ngắn hơn của hình ảnh có thể được sử dụng để tính `w` và `h` sử dụng `scale` và `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        ## tính toán box widths và heights cho toàn bộ aspect ratios
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                #tính anchor box thông thường khi aspect ratio = 1
                box_heigh = box_width = size * self.this_scale
                wh_list.append((box_width, box_heigh))
                if self.two_boxes_for_ar1:
                    #tính version lớn hơn của anchor box sử dụng the geometric mean của scale và next scale.
                    box_heigh = box_width = size * np.sqrt(self.this_scale * self.next_scale)
                    wh_list.append((box_width, box_heigh))
            else:
                #tính anchor box với aspect ratio khác 1
                box_width = int(size * self.this_scale * np.sqrt(ar))
                box_heigh = size * self.this_scale // np.sqrt(ar)
                wh_list.append((box_width, box_heigh))
        # append vào width height list
        wh_list = np.array(wh_list, dtype=np.float32)

        # Định hình input shape
        if K.common.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = x.get_shape().as_list()
        else:
            batch_size, feature_map_channels, feature_map_height, feature_map_width = x.get_shape().as_list()

        # Tính các center points của grid of box. Chúng là duy nhất đối với các aspect ratios.
        # Bước 2: Tính các step size. Khoảng cách là bao xa giữa các anchor box center point theo chiều width và height 
        if self.this_steps is None:
            step_x = self.img_width / feature_map_width
            step_y = self.img_height / feature_map_height
        else:
            if isinstance(self.this_steps, (list, tuple)) and len(self.this_steps) == 2:
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = step_width = self.this_steps
        # tính toán các offsets cho anchor box center point đầu tiên từ góc trên cùng bên trái của hình ảnh
        if self.this_offsets is None:
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and len(self.this_offsets) == 2:
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = offset_width = self.this_offsets
        # Bước 3: tính toán các tọa độ của (cx, cy, w, h) theo tọa độ của image gốc
        # bây giờ chúng ta có các offsets và step sizes, tính grid của anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        # tạo một 4D tensor có shape `(feature_map_height, feature_map_width, n_boxes, 4)` chứa tọa độ của center points của anchor boxes
        # chiều cuối cùng sẽ chứa tọa độ `(cx, cy, w, h)` của anchor boxes.
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes))  # đặt cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes))  # đặt cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # đặt w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # đặt h

        # Chuyển đổi tọa độ từ (cx, cy, w, h) sang (xmin, xmax, ymin, ymax)
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')
         
        # Nếu `clip_boxes` = True, giới hạn các tọa độ nằm trên boundary của hình ảnh
        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 2]] = x_coords
            y_coords = boxes_tensor[:, :, :, [1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [1, 3]] = y_coords

        # Nếu `normalize_coords` = True, chuẩn hóa tọa độ về khoảng [0,1] dựa trên kích thước của ảnh đầu vào
        if self.normalize_coords:
            boxes_tensor[:, :, :, 0] /= self.img_width
            boxes_tensor[:, :, :, 1] /= self.img_height
            boxes_tensor[:, :, :, 2] /= self.img_width
            boxes_tensor[:, :, :, 3] /= self.img_height

        if self.coords == 'centroids':
            # Chuyển đổi tọa độ về dạng (cx, cy, w, h)
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            # Chuyển đổi tọa độ về dạng (xmin, xmax, ymin, ymax)
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        # Tạo một tensor chứa các variances và apend vào `boxes_tensor`
        variances_tensor = np.zeros_like(boxes_tensor) #shape (feature_map_height, feature_map_width, n_boxes, 4)
        variances_tensor += self.variances # mở rộng variances_tensor với các giá trị variances
        # bây giờ `boxes_tensor` có shape (feature_map_height, feature_map_width, n_boxes, 8)
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)
        # Reshape boxes_tensor về dạng 5D tensor
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)  # thêm batch dimension
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))  # lặp lại cho mỗi batch

        return boxes_tensor
    
    def compute_output_shape(self, input_shape):
        if K.common.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        else: 
            batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'clip_boxes': self.clip_boxes,
            'variances': list(self.variances),
            'coords': self.coords,
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
# Test output of Anchor box
import tensorflow as tf
x = tf.random.normal(shape = (4, 38, 38, 512))

aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5]]
two_boxes_for_ar1=True
steps=[8, 16, 32, 64, 100, 300]
offsets=None
clip_boxes=False
variances=[0.1, 0.1, 0.2, 0.2]
coords='centroids'
normalize_coords=True
subtract_mean=[123, 117, 104]
divide_by_stddev=None
swap_channels=[2, 1, 0]
confidence_thresh=0.01
iou_threshold=0.45
top_k=200
nms_max_output_size=400


# Thiết lập tham số
img_height = 300
img_width = 300 
img_channels = 3 
mean_color = [123, 117, 104] 
swap_channels = [2, 1, 0] 
n_classes = 20 
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] 
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True


anchors = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2])(x)
print('anchors shape: ', anchors.get_shape())