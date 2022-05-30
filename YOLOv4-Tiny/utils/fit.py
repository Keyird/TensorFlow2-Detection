import tensorflow as tf
from tqdm import tqdm
from utils.utils import LossHistory

def fit_one_epoch(net, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, anchors,
                  num_classes, label_smoothing, regularization=False, train_step=None):
    loss = 0
    val_loss = 0
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, target0, target1 = batch[0], batch[1], batch[2]
            targets = [target0, target1]
            targets = [tf.convert_to_tensor(target) for target in targets]
            loss_value = train_step(images, yolo_loss, targets, net, optimizer, regularization, normalize)
            loss = loss + loss_value

            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1),
                                'lr': optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            # 计算验证集loss
            images, target0, target1 = batch[0], batch[1], batch[2]
            targets = [target0, target1]
            targets = [tf.convert_to_tensor(target) for target in targets]

            P5_output, P4_output = net(images)
            args = [P5_output, P4_output] + targets
            loss_value = yolo_loss(args, anchors, num_classes, label_smoothing=label_smoothing, normalize=normalize)
            if regularization:
                # 加入正则化损失
                loss_value = tf.reduce_sum(net.losses) + loss_value
            # 更新验证集loss
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)

    logs = {'loss': loss.numpy() / (epoch_size + 1), 'val_loss': val_loss.numpy() / (epoch_size_val + 1)}
    loss_history.on_epoch_end([], logs)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5' % (
    (epoch + 1), loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))


# 防止bug
def get_train_step_fn():
    @tf.function
    def train_step(imgs, yolo_loss, targets, net, optimizer, regularization, normalize):
        with tf.GradientTape() as tape:
            # 计算loss
            P5_output, P4_output = net(imgs, training=True)
            args = [P5_output, P4_output] + targets
            loss_value = yolo_loss(args, anchors, num_classes, label_smoothing=label_smoothing, normalize=normalize)
            if regularization:
                # 加入正则化损失
                loss_value = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value

    return train_step