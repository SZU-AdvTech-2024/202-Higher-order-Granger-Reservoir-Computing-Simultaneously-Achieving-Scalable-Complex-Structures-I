import numpy as np
from scipy import sparse
import torch
from scipy.sparse import csc_matrix,find
    
class Reservoir(object):
    """
    Build a reservoir and evaluate internal states
    
    """
    
    def __init__(self, n_internal_units=100, spectral_radius=0.99, leak=0.4, connectivity=0.3, input_scaling=0.2,
                 noise_level=0.01, edges_in=None, edges_ex=None, edges_out=None, args=None):
        '''
        n_internal_units=100：水库中的内部单元数量（即节点的数量）。
            spectral_radius=0.99：用于调整水库的特征值谱半径，控制水库的动态范围。（暂时设置）
            leak=0.4：泄漏参数，控制前一时刻和当前时刻的状态的加权。
            connectivity=0.3：内部节点之间的连接密度。
            input_scaling=0.2：输入权重的缩放因子。
            noise_level=0.01：用于加噪的噪声水平。
            edges_in, edges_ex, edges_out：表示水库中节点间的连接边（输入、输出、外部连接）。
            args=None：包含其他超参数的对象。
        '''

        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self.leak = leak
        self.edges_in = edges_in 
        self.edges_ex = edges_ex
        self.edges_out = edges_out
        self.args = args
        
        self.input_weights, self.rec, self.records = self._initialize_input_weights()#初始化输入权重
        
        # Generate internal weights 初始化RC储备池节点的连接权重
        self.internal_weights = self._initialize_internal_weights(
            n_internal_units,
            connectivity,
            spectral_radius)
    
    def conv(self, x):
        r=0
        rs=[]
        while(x!=0):
            r=x%2
            x=x//2
            rs=[r]+rs
        rs = rs[::-1]
        loc = []
        for i in range(len(rs)):
            if rs[i]==1:
                loc.append(i)
        return(np.array(loc))
    #为什么要保留位置为1的索引？？？？？
    
    def _initialize_input_weights(self):
        args, edges_in, edges_out, edges_ex = self.args, self.edges_in, self.edges_out, self.edges_ex 
        rec = []
        records = []
        input_weights = np.zeros((args.n*args.V*args.n_internal_units, args.n*args.V))
        
        for ni in range(args.n):
            sta = 0
            rec.append(sta)
            records.append(sta)
            input_weight = np.zeros((args.V*args.n_internal_units, args.n*args.V))
            us = []
            vs = []
            for edge in edges_in:
                us.append(self.conv(edge[0])+ni*args.V)
                vs.append(self.conv(edge[1])+ni*args.V) 
            for edge in edges_out:
                if edge[0]==ni:
                    for u in (self.conv(edges_ex[0,0])+ni*args.V):
                        us.append(u)
                        inout = self.conv(edges_ex[2,0])+edge[1]*args.V
                        if edges_ex[1,0]>=0:
                            inout = np.concatenate((inout,self.conv(edges_ex[1,0])+ni*args.V),axis=0)
                        vs.append(inout)
            
            for Vi in range(args.V):
                lov = Vi+ni*args.V
                vsi = []
                for i in range(len(us)):
                    if us[i]==lov:
                        vsi.append(vs[i])
                num = len(vsi)
        #         if num == 0:
        # # 如果 num 为 0，跳过后续的处理
        #             continue
                locs = []
                k = 0
                for i in range(num):
                    loc = []
                    for j in vsi[i]:
                        loc.append(j)
                    if len(loc)>0:
                        locs.append(loc)
                        k = k + len(loc)
                le = int(args.n_internal_units/num)
                
                for loc in locs:  
                    #weight = (2.0*np.random.binomial(1, 0.5, [le, len(loc)]) - 1.0)*args.input_scaling  
                    weight = (2.0*np.random.random(size=(le,len(loc))) - 1.0)*args.input_scaling
                    end = sta + le
                    input_weight[sta:end,loc] = weight
                    sta = end 
                    if loc == locs[-1]:
                        sta = (Vi+1)*args.n_internal_units
                    rec.append(sta)
                records.append(sta)
            input_weights[ni*args.V*args.n_internal_units:(ni+1)*args.V*args.n_internal_units,:] = input_weight
        row = []
        col = []
        data = []
        for i in range(input_weights.shape[0]):
            for j in range(input_weights.shape[1]):
                if input_weights[i,j] != 0:
                    row.append(i)
                    col.append(j)
                    data.append(input_weights[i,j])
        input_weights = csc_matrix((data, (row, col)), shape=(input_weights.shape[0], input_weights.shape[1]))
        return input_weights,rec,records
    
    def _initialize_internal_weights(self, n_internal_units, connectivity, spectral_radius):
        args, rec = self.args, self.rec
        V = args.V
        print("initialize_internal_weights")
        internal_weight = np.zeros((args.n*V*n_internal_units,args.n*V*n_internal_units))
        val = 0
        for i in range(len(rec)-1):
            sta = rec[i]
            end = rec[i+1]
            if end<sta:
                val = val+V*n_internal_units 
            else:
                e_max = 0
                while(e_max==0):
                    weight = sparse.rand(end-sta,end-sta,density=connectivity).todense() 
                    weight[np.where(weight > 0)] -= 0.5
                    # Adjust the spectral radius.
                    E, _ = np.linalg.eig(weight)
                    e_max = np.max(np.abs(E))
                    weight /= np.abs(e_max)/spectral_radius      
                internal_weight[sta+val:end+val,sta+val:end+val] = weight
                #print("sta-end",sta+val,end+val,weight.shape)
        row = []
        col = []
        data = []
        for k in range(args.n*args.V):
            for i in range(args.n_internal_units):
                for j in range(args.n_internal_units):
                    u = i+k*args.n_internal_units
                    v = j+k*args.n_internal_units
                    if internal_weight[u,v] != 0:
                        row.append(u)
                        col.append(v)
                        data.append(internal_weight[u,v])
        internal_weights = csc_matrix((data, (row, col)), shape=(internal_weight.shape[0], internal_weight.shape[1]))
        return internal_weights 
    
    def _compute_netx_state(self, previous_state, current_input):
        args, internal_weights, input_weights = self.args, self.internal_weights, self.input_weights 
        n = args.n
        V = args.V
        n_internal_units = args.n_internal_units 
        noise_level = args.noise_level
        leak = args.leak
        N, _ = previous_state.shape 
        
        state1 = previous_state 
        state2 = np.zeros((N, n*V*n_internal_units)) 
        state2 = internal_weights.dot(previous_state.T).T
        state2 += input_weights.dot(current_input.reshape(N,n*V).T).T
        state2 += np.random.rand(n*V*n_internal_units, N).T*noise_level + args.sigma
        state2 = np.tanh(state2)
        state_total = leak*state1 + (1-leak)*state2 
        return(state_total)
    
    def _compute_state_matrix(self, Xsn, n_drop=0):
        args, internal_weights, input_weights = self.args, self.internal_weights, self.input_weights 
        n_internal_units = args.n_internal_units 
        noise_level = args.noise_level
        leak = args.leak
        
        N, n, T, V = Xsn.shape 
        previous_state = np.zeros((N, n*V*n_internal_units)) 
        # Storage
        state_matrix = np.empty((N, T - n_drop, n*V*n_internal_units), dtype=float)
        for t in range(T):
            state1 = previous_state 
            state2 = np.zeros((N, n*V*n_internal_units)) 
            current_input = Xsn[:,:,t,:].reshape(N,n*V)
            state2 = internal_weights.dot(previous_state.T).T
            state2 += input_weights.dot(current_input.T).T
            state2 += np.random.rand(n*V*n_internal_units, N).T*noise_level + args.sigma
            state2 = np.tanh(state2)
            previous_state = leak*state1 + (1-leak)*state2
            # Store everything after the dropout period
            if (t > n_drop - 1):
                state_matrix[:, t - n_drop, :] = previous_state            
        return state_matrix
    
    def get_states(self, Xs, n_drop=0, bidir=True):
        
        N, n, T, V = Xs.shape
            
        # compute sequence of reservoir states
        states = self._compute_state_matrix(Xs, n_drop)
        
        # reservoir states on time reversed input
        if bidir is True:
            X_r = Xs[:, :, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = np.concatenate((states, states_r), axis=3)

        return states