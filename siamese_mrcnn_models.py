# Simaese Mask R-CNN Model

import tensorflow as tf
import sys
import os
import re
import time
import random
import numpy as np
import skimage.io
import imgaug

import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
import multiprocessing

MASK_RCNN_MODEL_PATH = 'Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
    
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize  

import utils as siamese_utils
import model as siamese_model


def pyramid_distance_graph(P, T, feature_maps=4, name='Tx'):
    L1P = []
    TA = KL.GlobalAveragePooling2D()(T)
    TM = KL.GlobalMaxPooling2D()(T)
    TA = KL.Lambda(lambda x: K.expand_dims(K.expand_dims(x, axis=1), axis=1))(TA)
    TM = KL.Lambda(lambda x: K.expand_dims(K.expand_dims(x, axis=1), axis=1))(TM)
    L1A = KL.Subtract()([P, TA])
    L1M = KL.Subtract()([P, TM])
    L1A = KL.Lambda(lambda x: K.abs(x))(L1A)
    L1M = KL.Lambda(lambda x: K.abs(x))(L1M)
#     NormP = KL.Lambda(lambda x: K.expand_dims(x, axis=-1))(P)
#     NormP = KL.GlobalAveragePooling3D()(NormP)
#     NormP = KL.Lambda(lambda x: K.expand_dims(K.expand_dims(K.abs(x) + 0.0001, axis=-1), axis=-1))(NormP)
    CCA = KL.Multiply()([P, TA])
    CCM = KL.Multiply()([P, TM])
#     CCA = KL.Lambda(lambda x: x / NormP)(CCA)
#     CCM = KL.Lambda(lambda x: x / NormP)(CCM)
#     DP = KL.Concatenate()([L1A, L1M])
#     def tile_and_abs(P, T):
#         out = K.tile(T, [1, int(P.shape[1]), int(P.shape[2]), 1])
#         out = K.abs(P-T)
#     L1P = KL.Lambda(lambda x: tile_and_abs(*x))([P, T])
    # Currently no 1x1 conv after L1
    L1A = KL.Conv2D(feature_maps, (1, 1), name='fpn_l1a_' + name)(L1A)
    L1M = KL.Conv2D(feature_maps, (1, 1), name='fpn_l1m_' + name)(L1M)
    CCA = KL.Conv2D(feature_maps, (1, 1), name='fpn_cca_' + name)(CCA)
    CCM = KL.Conv2D(feature_maps, (1, 1), name='fpn_ccm_' + name)(CCM)
    
    DP = KL.Concatenate()([L1A, L1M, CCA, CCM])
    
    return DP

def build_siamese_distance_model(feature_maps=20):
    # Define model to run resnet+fpn twice for image and target
    # Define inputs from target pyramid
    T2 = KL.Input(shape=[None,None,128], name="input_T2")
    T3 = KL.Input(shape=[None,None,128], name="input_T3")
    T4 = KL.Input(shape=[None,None,128], name="input_T4")
    T5 = KL.Input(shape=[None,None,128], name="input_T5")
    T6 = KL.Input(shape=[None,None,128], name="input_T6")
    P = KL.Input(shape=[None,None,128], name="input_P")
    # Compute L1 distances
    L1T2 = pyramid_distance_graph(P, T2, feature_maps=feature_maps, name='T2')
    L1T3 = pyramid_distance_graph(P, T3, feature_maps=feature_maps, name='T3')
    L1T4 = pyramid_distance_graph(P, T4, feature_maps=feature_maps, name='T4')
    L1T5 = pyramid_distance_graph(P, T5, feature_maps=feature_maps, name='T5')
    L1T6 = pyramid_distance_graph(P, T6, feature_maps=feature_maps, name='T6')
    out = KL.Concatenate(axis=-1)([L1T2, L1T3, L1T4, L1T5, L1T6])
    # Return model
    return KM.Model([P, T2, T3, T4, T5, T6], [out], name="fpn_l1_model")

def separable_conv(net, filter_size=15, feature_maps=128, name='separable_conv', train_bn=True):
    # Inspired by large kernels matter and light headed Faster R-CNN
    # Split into two paths
    A = B = net
    # Path A
    A = KL.Conv2D(feature_maps//2, (filter_size, 1), padding="SAME", name=name+"A1")(A)
    A = modellib.BatchNorm(name=name+'bn_A1')(A, training=train_bn)
    A = KL.Activation('relu')(A)
    A = KL.Conv2D(feature_maps, (1, filter_size), padding="SAME", name=name+"A2")(A)
    A = modellib.BatchNorm(name=name+'bn_A2')(A, training=train_bn)
    A = KL.Activation('relu')(A)
    # Path B
    B = KL.Conv2D(feature_maps//2, (1, filter_size), padding="SAME", name=name+"B1")(B)
    B = modellib.BatchNorm(name=name+'bn_B1')(A, training=train_bn)
    B = KL.Activation('relu')(A)
    B = KL.Conv2D(feature_maps, (filter_size, 1), padding="SAME", name=name+"B2")(B)
    B = modellib.BatchNorm(name=name+'bn_B2')(A, training=train_bn)
    B = KL.Activation('relu')(A)
    # Add Paths
    return KL.Add()([A, B])

def build_fpn_model(feature_maps=128):
    # Define model to run resnet+fpn twice for image and target
    # Define input image
    C2 = KL.Input(shape=[None,None,256], name="input_C2")
    C3 = KL.Input(shape=[None,None,512], name="input_C3")
    C4 = KL.Input(shape=[None,None,1024], name="input_C4")
    C5 = KL.Input(shape=[None,None,2048], name="input_C5")
#     LK2 = separable_conv(C2, filter_size=5, feature_maps=feature_maps, name='fpn_sep_c2', train_bn=True)
#     LK3 = separable_conv(C3, filter_size=7, feature_maps=feature_maps, name='fpn_sep_c3', train_bn=True)
#     LK4 = separable_conv(C4, filter_size=9, feature_maps=feature_maps, name='fpn_sep_c4', train_bn=True)
#     LK5 = separable_conv(C5, filter_size=11, feature_maps=feature_maps, name='fpn_sep_c5', train_bn=True)
    # Compute fpn activations
    P2, P3, P4, P5, P6 = fpn_graph(C2, C3, C4, C5, feature_maps=feature_maps)
    # Return model
    return KM.Model([C2, C3, C4, C5], [P2, P3, P4, P5, P6], name="fpn_model")


def fpn_graph(C2, C3, C4, C5, feature_maps=128):
    # CHANGE: Added featuremaps parameter
    P5 = KL.Conv2D(feature_maps, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(feature_maps, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(feature_maps, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(feature_maps, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(feature_maps, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(feature_maps, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(feature_maps, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(feature_maps, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    
    return [P2, P3, P4, P5, P6]

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layres
    Returns:
        logits: [N, NUM_CLASSES] classifier logits (before softmax)
        probs: [N, NUM_CLASSES] classifier probabilities
        bbox_deltas: [N, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_boxes, pool_height, pool_width, channels]
    x = modellib.PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(512, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(512, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layres
    Returns: Masks [batch, roi_count, height, width, num_classes]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = modellib.PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x

class FancySiameseMaskRCNN(siamese_model.SiameseMaskRCNN):
    """Encapsulates the Mask RCNN model functionality.
    The actual Keras model is in the keras_model property.
    """

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        # CHANGE: add target input
        input_target = KL.Input(
            shape=config.TARGET_SHAPE.tolist(), name="input_target")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: modellib.norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # CHANGE: Use weightshared FPN model for image and target
        # Create FPN Model
        resnet = siamese_model.build_resnet_model(self.config)
        fpn = build_fpn_model(feature_maps=self.config.FPN_FEATUREMAPS)
        siamese_distance = build_siamese_distance_model(feature_maps=self.config.FPN_FEATUREMAPS//32)
        # Create Image FP
        _, IC2, IC3, IC4, IC5 = resnet(input_image)
        IP2, IP3, IP4, IP5, IP6 = fpn([IC2, IC3, IC4, IC5])
        # Create Target FR
        _, TC2, TC3, TC4, TC5 = resnet(input_target)
        TP2, TP3, TP4, TP5, TP6 = fpn([TC2, TC3, TC4, TC5])
        
        # CHANGE: add siamese distance copmputation
        # Combine FPs using L1 distance
        DP2 = siamese_distance([IP2, TP2, TP3, TP4, TP5, TP6])
        DP3 = siamese_distance([IP3, TP2, TP3, TP4, TP5, TP6])
        DP4 = siamese_distance([IP4, TP2, TP3, TP4, TP5, TP6])
        DP5 = siamese_distance([IP5, TP2, TP3, TP4, TP5, TP6])
        DP6 = siamese_distance([IP6, TP2, TP3, TP4, TP5, TP6])
        
        DP2 = separable_conv(DP2, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS//2, name='fpn_sep_dp2', train_bn=self.config.TRAIN_BN)
        DP3 = separable_conv(DP3, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS//2, name='fpn_sep_dp3', train_bn=self.config.TRAIN_BN)
        DP4 = separable_conv(DP4, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS//2, name='fpn_sep_dp4', train_bn=self.config.TRAIN_BN)
        DP5 = separable_conv(DP5, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS//2, name='fpn_sep_dp5', train_bn=self.config.TRAIN_BN)
        DP6 = separable_conv(DP6, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS//2, name='fpn_sep_dp6', train_bn=self.config.TRAIN_BN)
        
        IP2 = separable_conv(IP2, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS, name='fpn_sep_ip2', train_bn=self.config.TRAIN_BN)
        IP3 = separable_conv(IP3, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS, name='fpn_sep_ip3', train_bn=self.config.TRAIN_BN)
        IP4 = separable_conv(IP4, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS, name='fpn_sep_ip4', train_bn=self.config.TRAIN_BN)
        IP5 = separable_conv(IP5, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS, name='fpn_sep_ip5', train_bn=self.config.TRAIN_BN)
        IP6 = separable_conv(IP6, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS, name='fpn_sep_ip6', train_bn=self.config.TRAIN_BN)
    
        # CHANGE: combine original and siamese features
        P2 = KL.Concatenate()([IP2, DP2])
        P3 = KL.Concatenate()([IP3, DP3])
        P4 = KL.Concatenate()([IP4, DP4])
        P5 = KL.Concatenate()([IP5, DP5])
        P6 = KL.Concatenate()([IP6, DP6])

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        # CHANGE: Set number of filters to 192 [128 original + 64 L1 + CC]
        rpn = modellib.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), (self.config.FPN_FEATUREMAPS + self.config.FPN_FEATUREMAPS//2))
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = modellib.ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: modellib.parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: modellig.norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask =\
                modellib.DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            # CHANGE: reduce number of classes to 2
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, num_classes=2,
                                     train_bn=config.TRAIN_BN)
            # CHANGE: reduce number of classes to 2
            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              num_classes=2,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: modellib.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: modellib.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            # CHANGE: use custom class loss without using active_class_ids
            class_loss = KL.Lambda(lambda x: siamese_model.mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: modellib.mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: modellib.mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            # CHANGE: Added target to inputs
            inputs = [input_image, input_image_meta, input_target,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            # CHANGE: reduce number of classes to 2
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, num_classes=2,
                                     train_bn=config.TRAIN_BN)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in 
            # normalized coordinates
            detections = modellib.DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            # CHANGE: reduce number of classes to 2
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              num_classes=2,
                                              train_bn=config.TRAIN_BN)

            # CHANGE: Added target to the input
            model = KM.Model([input_image, input_image_meta, input_target, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model
    
    

    
    
def new_fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layres
    Returns:
        logits: [N, NUM_CLASSES] classifier logits (before softmax)
        probs: [N, NUM_CLASSES] classifier probabilities
        bbox_deltas: [N, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_boxes, pool_height, pool_width, channels]
    x = modellib.PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(512, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(512, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    x = KL.Reshape((s[1],1, 4), name="mrcnn_bbox")(x)
    # Duplicate output for fg/bg detections
    mrcnn_bbox = KL.Concatenate(axis=-2)([x for i in range(num_classes)])

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def new_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layres
    Returns: Masks [batch, roi_count, height, width, num_classes]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = modellib.PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(1, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    # Duplicate output for fg/bg detections
    x = KL.Concatenate(axis=-1)([x for i in range(num_classes)])
    return x
    
    
def new_fpn_graph(C2, C3, C4, C5, feature_maps=128):
    # CHANGE: Added featuremaps parameter
    P5 = KL.Conv2D(feature_maps, (1, 1), activation="relu", name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(feature_maps, (1, 1), activation="relu", name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(feature_maps, (1, 1), activation="relu", name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(feature_maps, (1, 1), activation="relu", name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(feature_maps, (3, 3), activation="relu", padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(feature_maps, (3, 3), activation="relu", padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(feature_maps, (3, 3), activation="relu", padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(feature_maps, (3, 3), activation="relu", padding="SAME", name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    return [P2, P3, P4, P5, P6]


def build_new_fpn_model(feature_maps=128):
    # Define model to run resnet+fpn twice for image and target
    # Define input image
    C2 = KL.Input(shape=[None,None,256], name="input_C2")
    C3 = KL.Input(shape=[None,None,512], name="input_C3")
    C4 = KL.Input(shape=[None,None,1024], name="input_C4")
    C5 = KL.Input(shape=[None,None,2048], name="input_C5")
#     LK2 = separable_conv(C2, filter_size=5, feature_maps=feature_maps, name='fpn_sep_c2', train_bn=True)
#     LK3 = separable_conv(C3, filter_size=7, feature_maps=feature_maps, name='fpn_sep_c3', train_bn=True)
#     LK4 = separable_conv(C4, filter_size=9, feature_maps=feature_maps, name='fpn_sep_c4', train_bn=True)
#     LK5 = separable_conv(C5, filter_size=11, feature_maps=feature_maps, name='fpn_sep_c5', train_bn=True)
    # Compute fpn activations
    P2, P3, P4, P5, P6 = new_fpn_graph(C2, C3, C4, C5, feature_maps=feature_maps)
    # Return model
    return KM.Model([C2, C3, C4, C5], [P2, P3, P4, P5, P6], name="fpn_model")

    
class NewSiameseMaskRCNN(siamese_model.SiameseMaskRCNN):
    """Encapsulates the Mask RCNN model functionality.
    The actual Keras model is in the keras_model property.
    """

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        # CHANGE: add target input
        input_target = KL.Input(
            shape=config.TARGET_SHAPE.tolist(), name="input_target")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: modellib.norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # CHANGE: Use weightshared FPN model for image and target
        # Create FPN Model
        resnet = siamese_model.build_resnet_model(self.config)
        fpn = build_new_fpn_model(feature_maps=self.config.FPN_FEATUREMAPS)
        siamese_distance = build_siamese_distance_model(feature_maps=self.config.FPN_FEATUREMAPS//32)
        # Create Image FP
        _, IC2, IC3, IC4, IC5 = resnet(input_image)
        IP2, IP3, IP4, IP5, IP6 = fpn([IC2, IC3, IC4, IC5])
        # Create Target FR
        _, TC2, TC3, TC4, TC5 = resnet(input_target)
        TP2, TP3, TP4, TP5, TP6 = fpn([TC2, TC3, TC4, TC5])
        
        # CHANGE: add siamese distance copmputation
        # Combine FPs using L1 distance
        DP2 = siamese_distance([IP2, TP2, TP3, TP4, TP5, TP6])
        DP3 = siamese_distance([IP3, TP2, TP3, TP4, TP5, TP6])
        DP4 = siamese_distance([IP4, TP2, TP3, TP4, TP5, TP6])
        DP5 = siamese_distance([IP5, TP2, TP3, TP4, TP5, TP6])
        DP6 = siamese_distance([IP6, TP2, TP3, TP4, TP5, TP6])
        
        DP2 = separable_conv(DP2, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS//2, name='fpn_sep_dp2', train_bn=self.config.TRAIN_BN)
        DP3 = separable_conv(DP3, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS//2, name='fpn_sep_dp3', train_bn=self.config.TRAIN_BN)
        DP4 = separable_conv(DP4, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS//2, name='fpn_sep_dp4', train_bn=self.config.TRAIN_BN)
        DP5 = separable_conv(DP5, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS//2, name='fpn_sep_dp5', train_bn=self.config.TRAIN_BN)
        DP6 = separable_conv(DP6, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS//2, name='fpn_sep_dp6', train_bn=self.config.TRAIN_BN)
        
        IP2 = separable_conv(IP2, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS, name='fpn_sep_ip2', train_bn=self.config.TRAIN_BN)
        IP3 = separable_conv(IP3, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS, name='fpn_sep_ip3', train_bn=self.config.TRAIN_BN)
        IP4 = separable_conv(IP4, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS, name='fpn_sep_ip4', train_bn=self.config.TRAIN_BN)
        IP5 = separable_conv(IP5, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS, name='fpn_sep_ip5', train_bn=self.config.TRAIN_BN)
        IP6 = separable_conv(IP6, filter_size=15, feature_maps=self.config.FPN_FEATUREMAPS, name='fpn_sep_ip6', train_bn=self.config.TRAIN_BN)
    
        # CHANGE: combine original and siamese features
        P2 = KL.Concatenate()([IP2, DP2])
        P3 = KL.Concatenate()([IP3, DP3])
        P4 = KL.Concatenate()([IP4, DP4])
        P5 = KL.Concatenate()([IP5, DP5])
        P6 = KL.Concatenate()([IP6, DP6])

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        # CHANGE: Set number of filters to 192 [128 original + 64 L1 + CC]
        rpn = modellib.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), (self.config.FPN_FEATUREMAPS + self.config.FPN_FEATUREMAPS//2))
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = modellib.ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: modellib.parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: modellig.norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask =\
                modellib.DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            # CHANGE: reduce number of classes to 2
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                new_fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, num_classes=2,
                                     train_bn=config.TRAIN_BN)
            # CHANGE: reduce number of classes to 2
            mrcnn_mask = new_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              num_classes=2,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: modellib.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: modellib.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            # CHANGE: use custom class loss without using active_class_ids
            class_loss = KL.Lambda(lambda x: siamese_model.mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: modellib.mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: modellib.mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            # CHANGE: Added target to inputs
            inputs = [input_image, input_image_meta, input_target,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            # CHANGE: reduce number of classes to 2
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, num_classes=2,
                                     train_bn=config.TRAIN_BN)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in 
            # normalized coordinates
            detections = modellib.DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            # CHANGE: reduce number of classes to 2
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              num_classes=2,
                                              train_bn=config.TRAIN_BN)

            # CHANGE: Added target to the input
            model = KM.Model([input_image, input_image_meta, input_target, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model


    
    
class NosepSiameseMaskRCNN(NewSiameseMaskRCNN):
    """Encapsulates the Mask RCNN model functionality.
    The actual Keras model is in the keras_model property.
    """

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        # CHANGE: add target input
        input_target = KL.Input(
            shape=config.TARGET_SHAPE.tolist(), name="input_target")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: modellib.norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # CHANGE: Use weightshared FPN model for image and target
        # Create FPN Model
        resnet = siamese_model.build_resnet_model(self.config)
        fpn = build_new_fpn_model(feature_maps=self.config.FPN_FEATUREMAPS)
        siamese_distance = build_siamese_distance_model(feature_maps=self.config.FPN_FEATUREMAPS//32)
        # Create Image FP
        _, IC2, IC3, IC4, IC5 = resnet(input_image)
        IP2, IP3, IP4, IP5, IP6 = fpn([IC2, IC3, IC4, IC5])
        # Create Target FR
        _, TC2, TC3, TC4, TC5 = resnet(input_target)
        TP2, TP3, TP4, TP5, TP6 = fpn([TC2, TC3, TC4, TC5])
        
        # CHANGE: add siamese distance copmputation
        # Combine FPs using L1 distance
        DP2 = siamese_distance([IP2, TP2, TP3, TP4, TP5, TP6])
        DP3 = siamese_distance([IP3, TP2, TP3, TP4, TP5, TP6])
        DP4 = siamese_distance([IP4, TP2, TP3, TP4, TP5, TP6])
        DP5 = siamese_distance([IP5, TP2, TP3, TP4, TP5, TP6])
        DP6 = siamese_distance([IP6, TP2, TP3, TP4, TP5, TP6])
        
        # CHANGE: combine original and siamese features
        P2 = KL.Concatenate()([IP2, DP2])
        P3 = KL.Concatenate()([IP3, DP3])
        P4 = KL.Concatenate()([IP4, DP4])
        P5 = KL.Concatenate()([IP5, DP5])
        P6 = KL.Concatenate()([IP6, DP6])

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        # CHANGE: Set number of filters to 208 [128 original + 5*4*4 L1 + CC]
        rpn = modellib.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), (self.config.FPN_FEATUREMAPS + 5*4*4))
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = modellib.ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: modellib.parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: modellig.norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask =\
                modellib.DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            # CHANGE: reduce number of classes to 2
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                new_fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, num_classes=2,
                                     train_bn=config.TRAIN_BN)
            # CHANGE: reduce number of classes to 2
            mrcnn_mask = new_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              num_classes=2,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: modellib.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: modellib.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            # CHANGE: use custom class loss without using active_class_ids
            class_loss = KL.Lambda(lambda x: siamese_model.mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: modellib.mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: modellib.mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            # CHANGE: Added target to inputs
            inputs = [input_image, input_image_meta, input_target,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            # CHANGE: reduce number of classes to 2
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, num_classes=2,
                                     train_bn=config.TRAIN_BN)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in 
            # normalized coordinates
            detections = modellib.DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            # CHANGE: reduce number of classes to 2
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              num_classes=2,
                                              train_bn=config.TRAIN_BN)

            # CHANGE: Added target to the input
            model = KM.Model([input_image, input_image_meta, input_target, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model
    

    
def simple_distance_graph(P, T, feature_maps=128, name='Tx'):
    T = KL.GlobalAveragePooling2D()(T)
    T = KL.Lambda(lambda x: K.expand_dims(K.expand_dims(x, axis=1), axis=1))(T)
    T = KL.Lambda(lambda x: K.tile(T, [1, int(P.shape[1]), int(P.shape[2]), 1]))(T)
    L1 = KL.Subtract()([P, T])
    L1 = KL.Lambda(lambda x: K.abs(x))(L1)
    D = KL.Concatenate()([P, T, L1])
    if feature_maps:
        D = KL.Conv2D(feature_maps, (1, 1), name='fpn_distance_' + name)(D)
    
    return D
    
class SimpleSiameseMaskRCNN(NewSiameseMaskRCNN):
    """Encapsulates the Mask RCNN model functionality.
    The actual Keras model is in the keras_model property.
    """

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        # CHANGE: add target input
        input_target = KL.Input(
            shape=config.TARGET_SHAPE.tolist(), name="input_target")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: modellib.norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # CHANGE: Use weightshared FPN model for image and target
        # Create FPN Model
        resnet = siamese_model.build_resnet_model(self.config)
        fpn = build_new_fpn_model(feature_maps=self.config.FPN_FEATUREMAPS)
        # Create Image FP
        _, IC2, IC3, IC4, IC5 = resnet(input_image)
        IP2, IP3, IP4, IP5, IP6 = fpn([IC2, IC3, IC4, IC5])
        # Create Target FR
        _, TC2, TC3, TC4, TC5 = resnet(input_target)
        TP2, TP3, TP4, TP5, TP6 = fpn([TC2, TC3, TC4, TC5])
        
        # CHANGE: add siamese distance copmputation
        # Combine FPs using L1 distance
        P2 = simple_distance_graph(IP2, TP2, feature_maps = 3*self.config.FPN_FEATUREMAPS//2, name='P2')
        P3 = simple_distance_graph(IP3, TP3, feature_maps = 3*self.config.FPN_FEATUREMAPS//2, name='P3')
        P4 = simple_distance_graph(IP4, TP4, feature_maps = 3*self.config.FPN_FEATUREMAPS//2, name='P4')
        P5 = simple_distance_graph(IP5, TP5, feature_maps = 3*self.config.FPN_FEATUREMAPS//2, name='P5')
        P6 = simple_distance_graph(IP6, TP6, feature_maps = 3*self.config.FPN_FEATUREMAPS//2, name='P6')

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        # CHANGE: Set number of filters to 192 [3*self.config.FPN_FEATUREMAPS//2]
        rpn = modellib.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), (3*self.config.FPN_FEATUREMAPS//2))
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = modellib.ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: modellib.parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: modellig.norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask =\
                modellib.DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            # CHANGE: reduce number of classes to 2
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                new_fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, num_classes=2,
                                     train_bn=config.TRAIN_BN)
            # CHANGE: reduce number of classes to 2
            mrcnn_mask = new_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              num_classes=2,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: modellib.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: modellib.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            # CHANGE: use custom class loss without using active_class_ids
            class_loss = KL.Lambda(lambda x: siamese_model.mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: modellib.mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: modellib.mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            # CHANGE: Added target to inputs
            inputs = [input_image, input_image_meta, input_target,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            # CHANGE: reduce number of classes to 2
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                new_fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, num_classes=2,
                                     train_bn=config.TRAIN_BN)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in 
            # normalized coordinates
            detections = modellib.DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            # CHANGE: reduce number of classes to 2
            mrcnn_mask = new_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              num_classes=2,
                                              train_bn=config.TRAIN_BN)

            # CHANGE: Added target to the input
            model = KM.Model([input_image, input_image_meta, input_target, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model
