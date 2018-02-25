import numpy as np
import matplotlib.pyplot as plt
import colorsys

def load(exp_name):
    test_images = np.load('%s/test_images.npy' % (exp_name))
    test_labels = np.load('%s/test_labels.npy' % (exp_name))
    reconstructed_images = np.load('%s/reconstructed_images.npy' % (exp_name))
    latent = np.load('%s/latent.npy' % (exp_name))
    pred_labels = np.load('%s/pred_labels.npy' % (exp_name))

    return test_images, test_labels, reconstructed_images, latent, pred_labels

def make_latent_plot(exp_name, fignum=1):
    test_images, test_labels, reconstructed_images, latent, pred_labels = load(exp_name)

    true_num = np.argmax(test_labels, 1)
    pred_num = np.argmax(pred_labels, 1)

    plt.figure(fignum)

    for sp in [1, 2]:
        if sp == 1:
            target_num = true_num
        elif sp == 2:
            target_num = pred_num

        for num in range(10):
            idx = np.where(target_num == num)

            plt.subplot(2, 1, sp)
            color = colorsys.hsv_to_rgb(num/10.0, 0.8, 0.8)
            plt.scatter(latent[idx, 0], latent[idx, 1], c=color, alpha = 0.3, s=10, label=num)

        plt.axis('equal')
        #plt.axis('off')
        plt.grid('on')
        plt.legend(markerscale=4.0)

        if sp == 1:
            plt.title('True Labels')
        elif sp == 2:
            plt.title('Predicted Labels')

def get_class_probs(scores):
    # Relative to max
    rel_exp = scores - np.expand_dims(np.max(scores, axis=1), axis=1)

    # Clip
    clipped_scores = np.clip(rel_exp, -30, 0)
    #print 'Scores Range', np.min(clipped_scores[:]), np.max(clipped_scores[:])

    # Denominator
    sum_exp = np.expand_dims(np.sum(np.exp(clipped_scores), axis=1), axis=1)

    # Probs
    p_class = np.exp(clipped_scores) / (sum_exp * np.ones((1, 10)))

    #print scores[0, :]
    #print clipped_scores[0, :]
    #print p_class[0, :]
    #raw_input()

    return p_class

def identify_bad_nums(exp_name, comp_exp_name):
    # Load
    test_images, test_labels, reconstructed_images, latent, pred_labels = load(exp_name)
    _, _, _, _, comp_pred_labels = load(comp_exp_name)

    # Compute class probabilities
    p_class = get_class_probs(pred_labels)
    comp_p_class = get_class_probs(comp_pred_labels)

    sum_p1 = np.sum(p_class, axis=1)
    sum_p2 = np.sum(comp_p_class, axis=1)

    #print 'Range', np.min(p_class[:]), np.max(p_class[:])
    #print 'Range', np.min(comp_p_class[:]), np.max(comp_p_class[:])

    #print 'Sum', np.min(sum_p1), np.max(sum_p1)
    #print 'Sum', np.min(sum_p2), np.max(sum_p2)

    # Get true labels
    true_num = np.argmax(test_labels, 1)

    best_idx = {}
    best_metric = {}

    for i in range(len(true_num)):
        tl = true_num[i]

        if not tl in best_metric:
            best_metric[tl] = 0.0
            best_idx[tl] = None

        #re = np.mean( (test_images[i] - reconstructed_images[i])**2 )

        p_c = p_class[i, tl]
        p_c = np.clip(p_c, 0.1, 0.9)

        c_p_c = comp_p_class[i, tl]
        c_p_c = np.clip(c_p_c, 0.1, 0.9)

        p_arash = (1 - p_c) * (c_p_c)

        # Find ones that are misclassified by traditional autoencoder
        if best_metric[tl] > p_arash:
            continue
        #if p_class[i, tl] > 0.5:
        #    continue

        # But correctly classified by SAR one we're comparing against
        #if comp_p_class[i, tl] < 0.5:
        #    continue

        #if not (p_class[i, tl] < 0.5 and comp_p_class[i, tl] > 0.5):
        #    continue

        best_metric[tl] = p_arash
        best_idx[tl] = i

    return best_idx

def identify_good_nums(exp_name):
    # Load
    test_images, test_labels, reconstructed_images, latent, pred_labels = load(exp_name)

    # Get true labels
    true_num = np.argmax(test_labels, 1)

    best_idx = {}
    best_metric = {}

    for i in range(len(true_num)):
        tl = true_num[i]

        if not tl in best_metric:
            best_metric[tl] = 9999.0
            best_idx[tl] = None

        re = np.mean( (test_images[i] - reconstructed_images[i])**2 )

        # Find ones that have good reconstruction error
        if best_metric[tl] < re:
            continue

        best_metric[tl] = re
        best_idx[tl] = i

    return best_idx

ae_first_name = 'ae_first'
sim_name = 'simultaneous_0.01000'
#sim_name = ['simultaneous_%7.5f' % (cw) for cw in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] ]
sim_name = ['simultaneous_%7.5f' % (cw) for cw in [1.0] ]
exp_names = [ae_first_name] + sim_name

bad_idx = identify_bad_nums(ae_first_name, sim_name[-1])
good_idx = identify_good_nums(ae_first_name)

test_images, test_labels, reconstructed_images, latent, pred_labels = load(ae_first_name)

n_classes = 10
n_exps = len(exp_names)

plt.figure(100)

for exp_idx, exp_name in enumerate(exp_names):
    test_images, test_labels, reconstructed_images, latent, pred_labels = load(exp_name)
    p_class = get_class_probs(pred_labels)

    make_latent_plot(exp_name, exp_idx)

    for tl in range(n_classes):
        for gb, idx_map in enumerate([bad_idx, good_idx]):
            idx = idx_map[tl]
            if idx == None:
                continue

            plt.figure(exp_idx)
            plt.subplot(2, 1, 1)
            plt.plot(latent[idx, 0], latent[idx, 1], 'ko', markersize=10.0, markeredgewidth=4.0, mfc='none')

            plt.subplot(2, 1, 2)
            plt.plot(latent[idx, 0], latent[idx, 1], 'ko', markersize=10.0, markeredgewidth=4.0, mfc='none')

            im = reconstructed_images[idx, :]
            im = np.reshape(im, [28, 28])

            plt.figure(100)
            n_rows = n_exps + 1
            n_cols = n_classes * 2
            p_idx = (tl*2 + gb + 1) + exp_idx * n_cols
            plt.subplot(n_rows, n_cols, p_idx)
            plt.imshow(im)
            plt.title('%5.3f' % (100.0 * p_class[idx, tl]))
            plt.axis('off')

            if exp_idx == 0:
                im = test_images[idx, :]
                im = np.reshape(im, [28, 28])

                p_idx = (tl*2 + gb + 1) + n_exps * n_cols
                plt.subplot(n_rows, n_cols, p_idx)
                plt.imshow(im)
                #plt.title(tl)
                plt.axis('off')


plt.show()
