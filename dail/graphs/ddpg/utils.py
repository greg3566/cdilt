import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian

def jacobian_loss(Y,X):
    J = batch_jacobian(Y,X,use_pfor=False)
    return tf.reduce_sum(tf.reduce_mean(tf.square(J),axis=0))

def expert_type(env_name):
    if 'reacher' in env_name:
        return 'dail'
    elif 'Ant' in env_name:
        return 'sac'

def load_policy_graph(state, env_name, d_):
    assert expert_type(env_name) != 'dail'
    if 'Antv4' in env_name:
       fpath = 'policy/'+env_name+'/'+env_name+'_s0/tf1_save1700'
    elif 'Antv5' in env_name:
       fpath = 'policy/' + env_name + '/' + env_name + '_s0/tf1_save1400'
    else:
        raise NotImplementedError
    # model_info = joblib.load(osp.join(fpath, 'model_info.pkl'))
    # print(model_info)
    # {'inputs': {'x': 'Placeholder:0', 'a': 'Placeholder_1:0'}, 'outputs': {'mu': 'main/mul:0', 'q2': 'main/q2/Squeeze:0', 'q1': 'main/q1/Squeeze:0', 'pi': 'main/mul_1:0'}}
    graph = tf.Graph()
    scope = 'actor/' + d_ + '/loaded_expert'
    with tf.Session(graph=graph) as sess:
        tf.saved_model.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            fpath,
            scope
        )
        graphDef = graph.as_graph_def()
        vars = tf.global_variables()
        saver = tf.train.Saver(vars)
        saver.save(sess, fpath+'/weights_'+d_+'.ckpt')
    tf.graph_util.import_graph_def(graphDef, name='', input_map={scope+"/Placeholder:0": state})
    graph = tf.get_default_graph()
    for v in vars:
        graph.add_to_collection('loaded_variables', graph.get_tensor_by_name(v.name))
    return graph.get_tensor_by_name(scope+'/main/mul:0')

def load_policy_weights(sess, env_name):
    if expert_type(env_name) == 'dail':
        return 0
    if 'Antv4' in env_name:
        fpath = 'policy/' + env_name + '/' + env_name + '_s0/tf1_save1700'
    elif 'Antv5' in env_name:
        fpath = 'policy/' + env_name + '/' + env_name + '_s0/tf1_save1400'
    else:
        raise NotImplementedError
    graph = tf.get_default_graph()
    for d_ in ['learner', 'expert']:
        vars = graph.get_collection('loaded_variables', scope='actor/' + d_ + '/loaded_expert')
        saver = tf.train.Saver(vars)
        saver.restore(sess, fpath+'/weights_'+d_+'.ckpt')
    return 1
