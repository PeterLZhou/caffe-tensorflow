from kaffe.tensorflow import Network

class GazeNet(Network):
    def setup(self):
        (self.feed('input_data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 1.99999994948e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 1.99999994948e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .conv(1, 1, 1, 1, 1, name='conv5_red'))

        (self.feed('input_face')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1_face')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1_face')
             .lrn(2, 1.99999994948e-05, 0.75, name='norm1_face')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2_face')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2_face')
             .lrn(2, 1.99999994948e-05, 0.75, name='norm2_face')
             .conv(3, 3, 384, 1, 1, name='conv3_face')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4_face')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5_face')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5_face')
             .fc(500, name='fc6_face'))

        (self.feed('input_eyes_grid')
             .flatten(name='eyes_grid_flat')
             .power(24.0, 0.0, name='eyes_grid_mult'))

        (self.feed('eyes_grid_mult', 
                   'fc6_face')
             .concat(3, name='face_input')
             .fc(400, name='fc7_face')
             .fc(200, name='fc8_face')
             .fc(169, relu=False, name='importance_no_sigmoid')
             .sigmoid(name='importance_map_prefilter')
             .reshape(1, 13, 13, 1, transpose=True, name='importance_map_reshape')
             .conv(3, 3, 1, 1, 1, relu=False, name='importance_map'))

        (self.feed('importance_map', 
                   'conv5_red')
             .multiply(name='fc_7')
             .fc(25, relu=False, name='fc_0_0'))

        (self.feed('fc_7')
             .fc(25, relu=False, name='fc_1_0'))

        (self.feed('fc_7')
             .fc(25, relu=False, name='fc_m1_0'))

        (self.feed('fc_7')
             .fc(25, relu=False, name='fc_0_1'))

        (self.feed('fc_7')
             .fc(25, relu=False, name='fc_0_m1'))
