# -*- coding: utf-8 -*-
import sys
sys.path.append('../')

import os
import tensorflow as tf
import numpy as np
import time
from glob import glob
import io_util_ae as ut
import wgan_vgg_module as modules
import matplotlib.pyplot as plt
from metric import compute_measure


class AE(object):
    def __init__(self, sess):
        self.sess = sess    
        # some params
        self.trunc_min = -160.0
        self.trunc_max = 240.0
        self.dcm_path = '/home/cuiyang/data/zhilin/aapm_data_all'
        self.test_patient_no = ['L067']
        self.checkpoint_dir = './ckpt_ae'
        self.alpha = 1e-5  # learning rate.
        self.LDCT_path = 'quarter_3mm'
        self.NDCT_path = 'full_3mm'
        self.whole_size = 512
        self.patch_size = 64
        self.img_channel = 1
        self.img_vmax = 240.0
        self.img_vmin = -160.0
        self.batch_size = 128
        self.model = 'AE'
        self.phase = 'test'
        self.num_iter = 200000
        ####patients folder name
        self.train_patient_no = [d.split('/')[-1] for d in glob(self.dcm_path + '/*') if ('zip' not in d) & (d.split('/')[-1] not in self.test_patient_no)]     
        self.test_patient_no = self.test_patient_no    


        #save directory
        self.p_info = '_'.join(self.test_patient_no)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.p_info)
        self.log_dir = os.path.join('.', 'logs',  self.p_info)
        print('directory check!!\ncheckpoint : {}\ntensorboard_logs : {}'.format(self.checkpoint_dir, self.log_dir))

        #### set modules (generator, discriminator, vgg net)
        self.vgg = modules.autoencoder

        
        """
        load images
        """
        print('data load... dicom -> numpy') 
        self.image_loader = ut.DCMDataLoader(self.dcm_path, self.LDCT_path, self.NDCT_path, \
             image_size = self.whole_size, patch_size = self.patch_size, depth = self.img_channel,
             image_max = self.img_vmax, image_min = self.img_vmin, batch_size = self.batch_size, model = self.model)
                                     
        self.test_image_loader = ut.DCMDataLoader(self.dcm_path, self.LDCT_path, self.NDCT_path,\
             image_size = self.whole_size, patch_size = self.patch_size, depth = self.img_channel,
             image_max = self.img_vmax, image_min = self.img_vmin, batch_size = self.batch_size, model = self.model)
        
        # z_i -> input , x_i -> target
        
        t1 = time.time()
        if self.phase == 'train':
            self.image_loader(self.train_patient_no)
            self.test_image_loader(self.test_patient_no)
            print('data load complete !!!, {}\nN_train : {}, N_test : {}'.format(time.time() - t1, len(self.image_loader.input_image_name), len(self.test_image_loader.input_image_name)))
            [self.z_i, self.x_i] = self.image_loader.input_pipeline(self.sess, self.patch_size, self.num_iter)
        else:
            self.test_image_loader(self.test_patient_no)
            print('data load complete !!!, {}, N_test : {}'.format(time.time() - t1, len(self.test_image_loader.input_image_name)))
            self.z_i = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, self.img_channel], name = 'whole_LDCT')
            self.x_i = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, self.img_channel], name = 'whole_NDCT')

        """
        build model
        """
        #### image placehold  (patch image, whole i mage)
        self.whole_input = tf.placeholder(tf.float32, [1, self.whole_size, self.whole_size, self.img_channel], name = 'whole_LDCT')

        #### generate & discriminate
        # #generated images
        # self.G_zi = self.g_net(self.z_i, reuse = False)
        # self.G_whole_zi = self.g_net(self.whole_z)

        # #discriminate
        # self.D_xi = self.d_net(self.x_i, reuse = False)
        # self.D_G_zi= self.d_net(self.G_zi)

        # #### loss define
        # #gradients penalty
        # self.epsilon = tf.random_uniform([], 0.0, 1.0)
        # self.x_hat = self.epsilon * self.x_i + (1 - self.epsilon) * self.G_zi
        # self.D_x_hat = self.d_net(self.x_hat)
        # self.grad_x_hat = tf.gradients(self.D_x_hat, self.x_hat)[0]
        # self.grad_x_hat_l2 = tf.sqrt(tf.reduce_sum(tf.square(self.grad_x_hat), axis=1))
        # self.gradient_penalty =  tf.square(self.grad_x_hat_l2 - 1.0)

        #perceptual loss
        # self.G_zi_3c = tf.concat([self.G_zi]*3, axis=3)
        # self.xi_3c = tf.concat([self.x_i]*3, axis=3)
        # [w, h, d] = self.G_zi_3c.get_shape().as_list()[1:]

        # TODO: 计算autoencoder中间结果的mse
        # mse2=tf.losses.mean_squared_error(a, b)
        self.output_i, self.output_inter = self.vgg(self.z_i, reuse=False)
        self.vgg_loss = tf.losses.mean_squared_error(self.output_i, self.x_i)
        self.whole_output, self.whole_inter = self.vgg(self.whole_input, reuse=True)

        # self.vgg_perc_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((self.vgg.extract_feature(self.G_zi_3c) -  self.vgg.extract_feature(self.xi_3c))))) / (w*h*d))

        # #discriminator loss(WGAN LOSS)
        # d_loss = tf.reduce_mean(self.D_G_zi) - tf.reduce_mean(self.D_xi) 
        # grad_penal =  args.lambda_ *tf.reduce_mean(self.gradient_penalty )
        # self.D_loss = d_loss +grad_penal
        # #generator loss
        # self.G_loss = args.lambda_1 * self.vgg_perc_loss - tf.reduce_mean(self.D_G_zi)


        #### variable list
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.ae_vars = [var for var in t_vars if 'AE' in var.name]

        """
        summary
        """
        #loss summary
        self.summary_vgg_perc_loss = tf.summary.scalar("autoencoder_loss", self.vgg_loss)
        # self.summary_d_loss_all = tf.summary.scalar("2_DiscriminatorLoss_WGAN", self.D_loss)
        # self.summary_d_loss_1 = tf.summary.scalar("3_D_loss_disc", d_loss)
        # self.summary_d_loss_2 = tf.summary.scalar("4_D_loss_gradient_penalty", grad_penal)
        # self.summary_g_loss = tf.summary.scalar("GeneratorLoss", self.G_loss)
        # self.summary_all_loss = tf.summary.merge([self.summary_vgg_perc_loss, self.summary_d_loss_all, self.summary_d_loss_1, self.summary_d_loss_2, self.summary_g_loss])
            
        #psnr summary
        # self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", ut.tf_psnr(self.whole_z, self.whole_x, 1), family = 'PSNR')  # 0 ~ 1
        # self.summary_psnr_result = tf.summary.scalar("2_psnr_output", ut.tf_psnr(self.whole_x, self.G_whole_zi, 1), family = 'PSNR')  # 0 ~ 1
        # self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, self.summary_psnr_result])
        
 
        #image summary
        # self.check_img_summary = tf.concat([tf.expand_dims(self.z_i[0], axis=0), \
        #                                     tf.expand_dims(self.x_i[0], axis=0), \
        #                                     tf.expand_dims(self.G_zi[0], axis=0)], axis = 2)        
        # self.summary_train_image = tf.summary.image('0_train_image', self.check_img_summary)                                    
        # self.whole_img_summary = tf.concat([self.whole_input, self.whole_output], axis = 2)
        # self.summary_image = tf.summary.image('1_whole_image', self.whole_img_summary)
        
        #### optimizer
        # self.d_adam, self.g_adam = None, None
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     # self.d_adam = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.D_loss, var_list = self.d_vars)
        #     # self.g_adam = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.G_loss, var_list = self.g_vars)
        #     self.d_adam = tf.train.RMSPropOptimizer(learning_rate = args.alpha).minimize(self.D_loss, var_list = self.d_vars)
        #     self.g_adam = tf.train.RMSPropOptimizer(learning_rate = args.alpha).minimize(self.G_loss, var_list = self.g_vars)
        self.ae_adam = None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ae_adam = tf.train.RMSPropOptimizer(learning_rate = self.alpha).minimize(self.vgg_loss, var_list = self.ae_vars)

        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)    

        print('--------------------------------------------\n# of parameters : {} '.\
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        
    def train(self, args):
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.start_step = 0
        if args.continue_train:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                
        print('Start point : iter : {}'.format(self.start_step))

        start_time = time.time()

        for t in range(self.start_step, args.num_iter):
            # ae update
            self.sess.run(self.ae_adam)

            #print point
            if (t+1) % args.print_freq == 0:
                #print loss & time 
                ae_loss, summary_str0 = self.sess.run([self.vgg_loss, self.summary_vgg_perc_loss])
                #training sample check
                self.writer.add_summary(summary_str0, t)

                print('Iter {} Time {} ae_loss {}'.format(t, time.time() - start_time, ae_loss))
                # self.check_sample(t)

            if (t+1) % args.save_freq == 0:
                self.save(t)

        self.image_loader.coord.request_stop()
        self.image_loader.coord.join(self.image_loader.enqueue_threads)

    #summary test sample image during training
    # def check_sample(self, t):
    #     #summary whole image'
    #     sltd_idx = np.random.choice(range(len(self.test_image_loader.input_images)))
    #     test_zi, test_xi = self.test_image_loader.input_images[sltd_idx], self.test_image_loader.input_images[sltd_idx]
    #
    #     whole_G_zi = self.sess.run(self.whole_output, feed_dict={self.whole_input: test_zi.reshape(self.whole_input.get_shape().as_list())})
    #
    #     summary_str1, summary_str2= self.sess.run([self.summary_image], \
    #                              feed_dict={self.whole_input : test_zi.reshape(self.whole_input.get_shape().as_list()), \
    #                                         self.whole_output : whole_G_zi.reshape(self.whole_output.get_shape().as_list()), \
    #                                         })
    #     self.writer.add_summary(summary_str1, t)
    #     self.writer.add_summary(summary_str2, t)

      
    def save(self, step):
        model_name = 'autoencoder' + ".model"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)


    def load(self):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            print(self.start_step)
            return True
        else:
            return False

    def denormalize_(self, image):
        image = image * (3072 + 1024) - 1024
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        # x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray,
                     vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray,
                     vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray,
                     vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join('./test/L067',
                               'result_{}.png'.format(fig_name)))
        plt.close()

    def test(self, args):
        self.sess.run(tf.global_variables_initializer())

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ## mk save dir (image & numpy file)    
        npy_save_dir = os.path.join(args.test_npy_save_dir, self.p_info)

        if not os.path.exists(npy_save_dir):
            os.makedirs(npy_save_dir)

        from PIL import Image
        ## test
        start_time = time.time()
        for idx in range(len(self.test_image_loader.input_images)):
            test_zi, test_xi = self.test_image_loader.input_images[idx], self.test_image_loader.input_images[idx]
            
            whole_pred = self.sess.run(self.whole_output, feed_dict={self.whole_input: test_zi.reshape(self.whole_input.get_shape().as_list())})
            print('DEBUG:', whole_pred.shape)
            whole_pred = np.squeeze(whole_pred)
            # ut.save_image(test_zi, test_xi, whole_G_zi, save_dir="../test/L067_tf/")

            test_zi = self.trunc(self.denormalize_(test_zi))
            test_xi = self.trunc(self.denormalize_(test_xi))
            whole_pred = self.trunc(self.denormalize_(whole_pred))

            origin_result, pred_result = compute_measure(test_zi, test_xi, whole_pred, self.sess,data_range=(self.trunc_max - self.trunc_min))
            # self.sess.run(origin_result)
            self.save_fig(test_zi,test_xi, whole_pred, str(idx), origin_result, pred_result)



            