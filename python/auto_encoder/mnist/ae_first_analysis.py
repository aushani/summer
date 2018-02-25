import numpy as np
import matplotlib.pyplot as plt
import colorsys

#exp_name = 'ae_first'
exp_name = 'simultaneous_0.01000'

test_images = np.load('%s/test_images.npy' % (exp_name))
test_labels = np.load('%s/test_labels.npy' % (exp_name))
reconstructed_images = np.load('%s/reconstructed_images.npy' % (exp_name))
latent = np.load('%s/latent.npy' % (exp_name))
pred_labels = np.load('%s/pred_labels.npy' % (exp_name))

sum_exp = np.expand_dims(np.sum(np.exp(pred_labels), axis=1), axis=1)
p_class = np.exp(pred_labels) / (sum_exp * np.ones((1, 10)))

true_num = np.argmax(test_labels, 1)
pred_num = np.argmax(p_class, 1)

wrong = true_num != pred_num
print np.sum(wrong), len(pred_num)

bad_p_plotted = {}
bad_idx_plotted = {}

plt.figure(3)
for num in range(10):
    idx = np.where(true_num == num)
    color = colorsys.hsv_to_rgb(num/10.0, 0.8, 0.8)
    plt.scatter(latent[idx, 0], latent[idx, 1], c=color, alpha = 0.3, s=10, label=num)

plt.axis('equal')
#plt.axis('off')
plt.grid('on')
plt.legend(markerscale=4.0)

for i in range(len(pred_num)):
    if not wrong[i]:
        continue

    tl = true_num[i]
    pl = pred_num[i]

    if not tl in bad_p_plotted:
        bad_p_plotted[tl] = 1.0

    if bad_p_plotted[tl] < p_class[i, tl] and p_class[i, tl] > 0.3:
        continue

    bad_p_plotted[tl] = p_class[i, tl]
    bad_idx_plotted[tl] = i

    im0 = test_images[i, :]
    im1 = reconstructed_images[i, :]

    im0 = np.reshape(im0, [28, 28])
    im1 = np.reshape(im1, [28, 28])

    plt.figure(1)

    plt.subplot(2, 10, tl + 1)
    plt.imshow(im0)
    plt.title(tl)
    plt.axis('off')

    plt.subplot(2, 10, tl + 11)
    plt.imshow(im1)
    plt.title(pl)
    plt.axis('off')

good_rc_plotted = {}
for i in range(len(pred_num)):
    if wrong[i]:
        continue

    tl = true_num[i]
    pl = pred_num[i]

    im0 = test_images[i, :]
    im1 = reconstructed_images[i, :]

    rc = np.mean((im0 - im1)**2)

    if not tl in good_rc_plotted:
        good_rc_plotted[tl] = 9999.9

    if good_rc_plotted[tl] < rc:
        continue

    good_rc_plotted[tl] = rc

    im0 = np.reshape(im0, [28, 28])
    im1 = np.reshape(im1, [28, 28])

    plt.figure(2)

    plt.subplot(2, 10, tl + 1)
    plt.imshow(im0)
    plt.title(tl)
    plt.axis('off')

    plt.subplot(2, 10, tl + 11)
    plt.imshow(im1)
    plt.title(pl)
    plt.axis('off')

plt.figure(3)
for i in bad_idx_plotted:
    idx = bad_idx_plotted[i]
    plt.plot(latent[idx, 0], latent[i, 1], 'kx', markersize=16.0, markeredgewidth=4.0)

plt.show()
