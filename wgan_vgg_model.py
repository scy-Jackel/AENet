# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import time
from glob import glob
import inout_util as ut
import wgan_vgg_module as modules
import matplotlib.pyplot as plt
from metric import compute_measure
import cv2

def denormAndTrunc(image, norm_min=-1024.0, norm_max=3072, trunc_min=-240.0, trunc_max=160.0):
    image = image * (norm_max - norm_min) + norm_min
    image[image <= trunc_min] = trunc_min
    image[image >= trunc_max] = trunc_max
    image = (image - trunc_min) / (trunc_max - trunc_min)
    image = image * 255
    return image


class wganVgg(object):
    def __init__(self, sess, args):
        self.sess = sess    
        # some params
        self.trunc_min = -160.0
        self.trunc_max = 240.0
        self.ae_model_path = './AE/ckpt_ae/L067_3072'
        self.ae_checkpoint_dir = './AE/ckpt_ae/L067_3072'
        
        ####patients folder name
        self.train_patient_no = [d.split('/')[-1] for d in glob(args.dcm_path + '/*') if ('zip' not in d) & (d.split('/')[-1] not in args.test_patient_no)]     
        self.test_patient_no = args.test_patient_no
        # self.test_patient_no = ['L067_noSobel']

        #save directory
        self.p_info = '_'.join(self.test_patient_no)
        # self.checkpoint_dir = os.path.join('.', args.checkpoint_dir, 'L067_noSobel')
        self.checkpoint_dir = os.path.join('.', args.checkpoint_dir, 'L506')
        self.log_dir = os.path.join('.', 'logs',  self.p_info)
        print('directory check!!\ncheckpoint : {}\ntensorboard_logs : {}'.format(self.checkpoint_dir, self.log_dir))

        #### set modules (generator, discriminator, vgg net)
        self.g_net = modules.generator
        self.d_net = modules.discriminator
        # self.vgg = modules.Vgg19(vgg_path = args.pretrained_vgg)
        self.vgg = modules.AE(ae_path = './autoencoder_3072.npy')
        # vgg_session = tf.Session()
        # self.load_ae(vgg_session)

        
        """
        load images
        """
        print('data load... dicom -> numpy') 
        self.image_loader = ut.DCMDataLoader(args.dcm_path, args.LDCT_path, args.NDCT_path, \
             image_size = args.whole_size, patch_size = args.patch_size, depth = args.img_channel,
             image_max = args.img_vmax, image_min = args.img_vmin, batch_size = args.batch_size, model = args.model)
                                     
        self.test_image_loader = ut.DCMDataLoader(args.dcm_path, args.LDCT_path, args.NDCT_path,\
             image_size = args.whole_size, patch_size = args.patch_size, depth = args.img_channel,
             image_max = args.img_vmax, image_min = args.img_vmin, batch_size = args.batch_size, model = args.model)

        t1 = time.time()
        if args.phase == 'train':
            self.image_loader(self.train_patient_no)
            self.test_image_loader(self.test_patient_no)
            print('data load complete !!!, {}\nN_train : {}, N_test : {}'.format(time.time() - t1, len(self.image_loader.LDCT_image_name), len(self.test_image_loader.LDCT_image_name)))
            [self.z_i, self.x_i] = self.image_loader.input_pipeline(self.sess, args.patch_size, args.num_iter)
        else:
            self.test_image_loader(self.test_patient_no)
            print('data load complete !!!, {}, N_test : {}'.format(time.time() - t1, len(self.test_image_loader.LDCT_image_name)))
            self.z_i = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'whole_LDCT')
            self.x_i = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'whole_LDCT')

        """
        build model
        """
        #### image placehold  (patch image, whole image)
        self.whole_z = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_LDCT')
        self.whole_x = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_NDCT')

        #### generate & discriminate
        #generated images
        self.G_zi = self.g_net(self.z_i, reuse = False)
        self.G_whole_zi = self.g_net(self.whole_z)

        #discriminate
        self.D_xi = self.d_net(self.x_i, reuse = False)
        self.D_G_zi= self.d_net(self.G_zi)

        #### loss define
        #gradients penalty
        self.epsilon = tf.random_uniform([], 0.0, 1.0)
        self.x_hat = self.epsilon * self.x_i + (1 - self.epsilon) * self.G_zi
        self.D_x_hat = self.d_net(self.x_hat)
        self.grad_x_hat = tf.gradients(self.D_x_hat, self.x_hat)[0]
        self.grad_x_hat_l2 = tf.sqrt(tf.reduce_sum(tf.square(self.grad_x_hat), axis=1))
        self.gradient_penalty =  tf.square(self.grad_x_hat_l2 - 1.0)

        #perceptual loss
        self.G_zi_3c = tf.concat([self.G_zi]*3, axis=3)
        self.xi_3c = tf.concat([self.x_i]*3, axis=3)
        [w, h, d] = self.G_zi_3c.get_shape().as_list()[1:]

        self.inter_g = self.vgg.extract_feature(self.G_zi)
        self.inter_x = self.vgg.extract_feature(self.x_i)
        self.vgg_perc_loss = tf.losses.mean_squared_error(self.inter_g, self.inter_x)
        self.edge_cost = tf.reduce_mean(tf.squared_difference(tf.image.sobel_edges(self.G_zi), tf.image.sobel_edges(self.x_i)))
        # self.vgg_perc_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((self.vgg.extract_feature(self.G_zi_3c) -  self.vgg.extract_feature(self.xi_3c))))) / (w*h*d))

        #discriminator loss(WGAN LOSS)
        d_loss = tf.reduce_mean(self.D_G_zi) - tf.reduce_mean(self.D_xi) 
        grad_penal =  args.lambda_ *tf.reduce_mean(self.gradient_penalty )
        self.D_loss = d_loss + grad_penal
        #generator loss
        self.G_loss = args.lambda_1 * (self.vgg_perc_loss+self.edge_cost) - tf.reduce_mean(self.D_G_zi)


        #### variable list
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        """
        summary
        """
        #loss summary
        self.summary_vgg_perc_loss = tf.summary.scalar("1_PerceptualLoss_VGG", self.vgg_perc_loss)
        self.summary_d_loss_all = tf.summary.scalar("2_DiscriminatorLoss_WGAN", self.D_loss)
        self.summary_d_loss_1 = tf.summary.scalar("3_D_loss_disc", d_loss)
        self.summary_d_loss_2 = tf.summary.scalar("4_D_loss_gradient_penalty", grad_penal)
        self.summary_g_loss = tf.summary.scalar("GeneratorLoss", self.G_loss)
        self.summary_edge_cost = tf.summary.scalar("EdgeCost", self.edge_cost)
        self.summary_all_loss = tf.summary.merge([self.summary_vgg_perc_loss, self.summary_edge_cost, self.summary_d_loss_all, self.summary_d_loss_1, self.summary_d_loss_2, self.summary_g_loss])
            
        #psnr summary
        self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", ut.tf_psnr(self.whole_z, self.whole_x, 1), family = 'PSNR')  # 0 ~ 1
        self.summary_psnr_result = tf.summary.scalar("2_psnr_output", ut.tf_psnr(self.whole_x, self.G_whole_zi, 1), family = 'PSNR')  # 0 ~ 1
        self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, self.summary_psnr_result])
        
 
        #image summary
        self.check_img_summary = tf.concat([tf.expand_dims(self.z_i[0], axis=0), \
                                            tf.expand_dims(self.x_i[0], axis=0), \
                                            tf.expand_dims(self.G_zi[0], axis=0)], axis = 2)        
        self.summary_train_image = tf.summary.image('0_train_image', self.check_img_summary)                                    
        self.whole_img_summary = tf.concat([self.whole_z, self.whole_x, self.G_whole_zi], axis = 2)        
        self.summary_image = tf.summary.image('1_whole_image', self.whole_img_summary)
        
        #### optimizer
        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # self.d_adam = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.D_loss, var_list = self.d_vars)
            # self.g_adam = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.G_loss, var_list = self.g_vars)
            self.d_adam = tf.train.RMSPropOptimizer(learning_rate = args.alpha).minimize(self.D_loss, var_list = self.d_vars)
            self.g_adam = tf.train.RMSPropOptimizer(learning_rate = args.alpha).minimize(self.G_loss, var_list = self.g_vars)

        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)    

        print('--------------------------------------------\n# of parameters : {} '.\
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        
    def train(self, args):
        self.sess.run(tf.global_variables_initializer())

        # ret = self.load_ae()
        # assert ret==True

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
            for _ in range(0, args.d_iters):

                #discriminator update
                self.sess.run(self.d_adam)
 
            #generator update & loss summary
            _, summary_str= self.sess.run([self.g_adam, self.summary_all_loss])
            self.writer.add_summary(summary_str, t)

            #print point
            if (t+1) % args.print_freq == 0:
                #print loss & time 
                d_loss, g_loss, g_zi_img, summary_str0 = self.sess.run([self.D_loss, self.G_loss, self.G_zi, self.summary_train_image])
                #training sample check
                self.writer.add_summary(summary_str0, t)

                print('Iter {} Time {} d_loss {} g_loss {}'.format(t, time.time() - start_time, d_loss, g_loss))
                self.check_sample(args, t)

            if (t+1) % args.save_freq == 0:
                self.save(args, t)

        self.image_loader.coord.request_stop()
        self.image_loader.coord.join(self.image_loader.enqueue_threads)

    #summary test sample image during training
    def check_sample(self, args, t):
        #summary whole image'
        sltd_idx = np.random.choice(range(len(self.test_image_loader.LDCT_images)))
        test_zi, test_xi = self.test_image_loader.LDCT_images[sltd_idx], self.test_image_loader.NDCT_images[sltd_idx]

        whole_G_zi = self.sess.run(self.G_whole_zi, feed_dict={self.whole_z: test_zi.reshape(self.whole_z.get_shape().as_list())})
        
        summary_str1, summary_str2= self.sess.run([self.summary_image, self.summary_psnr], \
                                 feed_dict={self.whole_z : test_zi.reshape(self.whole_z.get_shape().as_list()), \
                                            self.whole_x : test_xi.reshape(self.whole_x.get_shape().as_list()), \
                                            self.G_whole_zi : whole_G_zi.reshape(self.G_whole_zi.get_shape().as_list()),
                                            })
        self.writer.add_summary(summary_str1, t)
        self.writer.add_summary(summary_str2, t)

      
    def save(self, args, step):
        model_name = args.model + ".model"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)

    def load_ae(self, session):
        print("loading auto encoder.....")
        ckpt = tf.train.get_checkpoint_state(self.ae_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(ckpt_name.split('-')[-1])
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
            saver.restore(session, os.path.join(self.ae_checkpoint_dir, ckpt_name))
            print(self.start_step)
            return True
        else:
            print('LOAD ERROR!!!!')
            return False


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
        # f, ax = plt.subplots(1, 3, figsize=(30, 10))
        # ax[0].imshow(x, cmap=plt.cm.gray)
        # ax[0].set_title('Quarter-dose', fontsize=30)
        # ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
        #                                                                    original_result[1],
        #                                                                    original_result[2]), fontsize=20)
        # ax[1].imshow(pred, cmap=plt.cm.gray)
        # ax[1].set_title('Result', fontsize=30)
        # ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
        #                                                                    pred_result[1],
        #                                                                    pred_result[2]), fontsize=20)
        # ax[2].imshow(y, cmap=plt.cm.gray)
        # ax[2].set_title('Full-dose', fontsize=30)

        # f.savefig(os.path.join('../test/L506' ,
        #                        'result_{}.jpg'.format(fig_name)))
        # plt.close()

        # print('save_path:', os.path.join('../test/L506_ae_split' ,'{}_PRED.jpg'.format(fig_name)))
        # cv2.imwrite(os.path.join('../test/L506_ae_split' ,'{}_PRED.jpg'.format(fig_name)),pred)

        print('save_path:', os.path.join('../test/L506_aeSobel_split' ,'{}_PRED.jpg'.format(fig_name)))
        cv2.imwrite(os.path.join('../test/L506_aeSobel_split' ,'{}_PRED.jpg'.format(fig_name)),pred)

    def test(self, args):
        self.sess.run(tf.global_variables_initializer())

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ## mk save dir (image & numpy file)    
        npy_save_dir = os.path.join('.', args.test_npy_save_dir, self.p_info)

        if not os.path.exists(npy_save_dir):
            os.makedirs(npy_save_dir)

        from PIL import Image
        ## test
        start_time = time.time()

        psnr_ldct = []
        ssim_ldct = []
        rmse_ldct = []
        psnr_pred = []
        ssim_pred = []
        rmse_pred= []

        for idx in range(len(self.test_image_loader.LDCT_images)):
            test_zi, test_xi = self.test_image_loader.LDCT_images[idx], self.test_image_loader.NDCT_images[idx]
            
            whole_G_zi = self.sess.run(self.G_whole_zi, feed_dict={self.whole_z: test_zi.reshape(self.whole_z.get_shape().as_list())})
            
            save_file_nm_f = 'from_' +  self.test_image_loader.LDCT_image_name[idx]
            save_file_nm_t = 'to_' +  self.test_image_loader.NDCT_image_name[idx]
            save_file_nm_g = 'Gen_from_' +  self.test_image_loader.LDCT_image_name[idx]
            
            # np.save(os.path.join(npy_save_dir, save_file_nm_f), test_zi)
            # np.save(os.path.join(npy_save_dir, save_file_nm_t), test_xi)
            # np.save(os.path.join(npy_save_dir, save_file_nm_g), whole_G_zi)

            whole_G_zi = np.squeeze(whole_G_zi)
            # ut.save_image(test_zi, test_xi, whole_G_zi, save_dir="../test/L067_tf/")

            # test_xi = self.trunc(self.denormalize_(test_xi))
            # test_zi = self.trunc(self.denormalize_(test_zi))
            # whole_G_zi = self.trunc(self.denormalize_(whole_G_zi))
            test_xi = denormAndTrunc(test_xi)
            test_zi = denormAndTrunc(test_zi)
            whole_G_zi = denormAndTrunc(whole_G_zi)

            # origin_result, pred_result = compute_measure(test_zi, test_xi, whole_G_zi, self.sess,data_range=1)
            origin_result, pred_result = compute_measure(test_zi, test_xi, whole_G_zi, self.sess,data_range=255)

            psnr_ldct.append(origin_result[0])
            ssim_ldct.append(origin_result[1])
            rmse_ldct.append(origin_result[2])
            psnr_pred.append(pred_result[0])
            ssim_pred.append(pred_result[1])
            rmse_pred.append(pred_result[2])


            self.save_fig(test_zi,test_xi, whole_G_zi, str(idx), origin_result, pred_result)
            print(idx, 'ok')

        print("ldct average psnr:{} , psnr std:{}".format(np.mean(psnr_ldct), np.std(psnr_ldct)))
        print("ldct average ssim:{} , ssim std:{}".format(np.mean(ssim_ldct), np.std(ssim_ldct)))
        print("pred average psnr:{} , psnr std:{}".format(np.mean(psnr_pred), np.std(psnr_pred)))
        print("pred average ssim:{} , ssim std:{}".format(np.mean(ssim_pred), np.std(ssim_pred)))

            