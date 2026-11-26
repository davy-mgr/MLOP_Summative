import tensorflow as tf


def augment_spectrogram(x, y):
    mel_bins, time_steps = tf.shape(x)[0], tf.shape(x)[1]

    x = tf.roll(
        x,
        tf.random.uniform(
            [],
            -(time_steps // 5),
            time_steps // 5 + 1,
            dtype=tf.int32
        ),
        axis=1
    )

    freq_width = tf.random.uniform([], 0, 9, dtype=tf.int32)
    freq_start = tf.random.uniform(
        [], 0, mel_bins - freq_width, dtype=tf.int32
    )

    time_width = tf.random.uniform([], 0, 31, dtype=tf.int32)
    time_start = tf.random.uniform(
        [], 0, time_steps - time_width, dtype=tf.int32
    )

    freq_mask = (
        (tf.range(mel_bins) >= freq_start) &
        (tf.range(mel_bins) < freq_start + freq_width)
    )[:, None, None]

    time_mask = (
        (tf.range(time_steps) >= time_start) &
        (tf.range(time_steps) < time_start + time_width)
    )[None, :, None]

    x = tf.where(freq_mask | time_mask, tf.zeros_like(x), x)

    return x + tf.random.normal(
        tf.shape(x), stddev=0.05
    ), y
