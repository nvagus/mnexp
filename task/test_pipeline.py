# -*- coding: utf-8 -*-
import utils
import document
import numpy as np
import keras.backend as K
import keras
import settings

class TestPipeline:
    def __init__(self, config: settings.Config):
        self.config = config

    def load_model(self):
        paths = self.config.model_output
        self.model = utils.load_model(paths)
        self.score_encoder = None
        
    def _test_doc_vec_gen(self):
        docs_path = self.config.pipeline_inputs[0]
        with utils.open(docs_path) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                doc = line[0]
                inputs = line[1]
                yield doc,inputs
        
    def get_doc_parser(self):
        doc_encoder = self.model.get_layer('doc_encoder')
        doc_input_shape = doc_encoder.layers[0].input_shape[-1]
        
        title_parser = document.DocumentParser(
                    document.parse_document(),
                    document.pad_document(1, doc_input_shape)
        )
        return title_parser
    
    def test_doc_vec(self):
        batch_size = self.config.batch_size
        docs = []
        bc_inputs = []
        cnt = 0
        doc_vec = {}
        doc_encoder = self.model.get_layer('doc_encoder')
        doc_parser = self.get_doc_parser()
        for doc,inputs in self._test_doc_vec_gen():  
            docs.append(doc)
            
            bc_inputs.append(doc_parser(inputs))
            cnt = cnt + 1
            if cnt == batch_size:
                outputs = doc_encoder.predict(np.squeeze(np.stack(bc_inputs)))
#                final_outputs = final_doc_encoder.predict(outputs)
#                for d,o,f in zip(docs,outputs,final_outputs):
#                    doc_vec[d] = o
#                    final_doc_vec[d] = f
                for d,o in zip(docs,outputs):
                    doc_vec[d] = o
                docs = []
                bc_inputs = []
                cnt = 0
        
        if cnt != 0:
            outputs = doc_encoder.predict(np.squeeze(np.stack(bc_inputs)))
            
            for d,o in zip(docs,outputs):
                doc_vec[d] = o
            docs = []
            bc_inputs = []
            cnt = 0    
        
        self.doc_vec = doc_vec


    def _test_user_vec_gen(self):
        users_path = self.config.pipeline_inputs[1]
        with utils.open(users_path) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                user = line[0] + line[1]
                clicks = line[2].split('#N#')
                yield user,clicks

    
    def test_user_vec(self):
        user_vec = {}

        user_encoder = self.model.get_layer('user_encoder')
        user_clicked_vec_shape = user_encoder.get_layer('user_clicked_vec').input_shape
        batch_size = self.config.batch_size
                
        users = []
        clicked_vec = []
        cnt = 0
        doc_vec = self.doc_vec
        
        undoc = set()
        for user,inputs in self._test_user_vec_gen():  
            users.append(user)
            
            vecs = np.zeros(user_clicked_vec_shape[1:])
            length = len(inputs)
            if length > user_clicked_vec_shape[1]:
                length = user_clicked_vec_shape[1]
            for i in range(-1,-1-length,-1):
                inp = inputs[i]
                if inp in doc_vec:
                    vecs[i,:] = doc_vec[inp]
                else:
                    undoc.add(inp)
            
            clicked_vec.append(vecs)
            cnt = cnt + 1
            
            if cnt == batch_size:
                outputs = user_encoder.predict(np.stack(clicked_vec))
                for u,o in zip(users,outputs):
                    user_vec[u] = o
                users = []
                clicked_vec = []
                cnt = 0
        
        if cnt != 0:
            outputs = user_encoder.predict(np.stack(clicked_vec))
            for u,o in zip(users,outputs):
                user_vec[u] = o
            users = []
            clicked_vec = []
            cnt = 0
        print(len(undoc))
        self.user_vec = user_vec    

    def get_score_encoder(self):
        if self.score_encoder is not None:
            return self.score_encoder
        
        model = self.model
        input1 = keras.layers.Input(model.get_layer('user_encoder').output_shape[1:])
        input2 = keras.layers.Input(model.get_layer('doc_encoder').output_shape[1:])
        join_vec = keras.layers.concatenate([input1, input2])
        
        concat_dense = model.get_layer('concat_dense')
        hidden = concat_dense(join_vec)
        
        W_score = model.get_layer('socre_dense').weights
        W_score = [K.eval(W) for W in W_score]
        score_dense = keras.layers.Dense(1,weights=W_score)
        score = score_dense(hidden)
        self.score_encoder = keras.Model([input1,input2],score)
        return self.score_encoder
    
    def _test_user_doc_gen(self):
        pair_path =  self.config.pipeline_inputs[2]
        with utils.open(pair_path) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                user_id = line[0]
                user_type = line[1]
                doc = line[2]
                yield user_id,user_type,doc
                
    def test_user_doc_score(self):
        pair_cnt = 0
        real_cnt = 0
        cnt = 0
        out_path = self.config.pipeline_output
        batch_size = self.config.batch_size
        
        users = []
        docs = []
        users_vec = []
        docs_vec = []
        
        user_vec = self.user_vec
        doc_vec = self.doc_vec
        score_encoder = self.get_score_encoder()
        with utils.open(out_path,'w') as ff:
            for user_id,user_type,doc in self._test_user_doc_gen():  
                pair_cnt = pair_cnt +1 
                user = user_id + user_type
                if user in user_vec and doc in doc_vec:
                    users.append((user_id,user_type))
                    docs.append(doc)
                    users_vec.append(user_vec[user])
                    docs_vec.append(doc_vec[doc])
                    real_cnt = real_cnt + 1
                    cnt = cnt + 1
                    
                    if cnt == batch_size:
                        outputs = score_encoder.predict([np.stack(users_vec),np.stack(docs_vec)])
                        
                        for idtp,do,out in zip(users,docs,outputs):
                            ff.write(idtp[0]+'\t'+idtp[1]+'\t'+do+'\t'+str(out[0])+'\n')
                            
                        users = []
                        docs = []
                        users_vec = []
                        docs_vec = []
                        cnt = 0
                
                
            if cnt != 0:
                outputs = score_encoder.predict([np.stack(users_vec),np.stack(docs_vec)])
                        
                for idtp,do,out in zip(users,docs,outputs):
                    ff.write(idtp[0]+'\t'+idtp[1]+'\t'+do+'\t'+str(out[0])+'\n')
                    
                users = []
                docs = []
                users_vec = []
                docs_vec = []
                cnt = 0
                
    
    def test_correct(self):
        myuser = None 
        mydoc = None
        
        doc2title = {}
        doc_parser = self.get_doc_parser()
        
        for doc,inputs in self._test_doc_vec_gen():
            doc2title[doc] = doc_parser(inputs)
        
        users_path =  self.config.pipeline_inputs[1]
        with open(users_path) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                user = line[0] + line[1]
                myuser = user
                inputs = line[2].split('#N#')
                break
        
        doc_encoder = self.model.get_layer('doc_encoder')
        doc_input_shape = doc_encoder.layers[0].input_shape[1:]
        
        user_encoder = self.model.get_layer('user_encoder')
        user_clicked_vec_shape = user_encoder.get_layer('user_clicked_vec').input_shape
        docs = []
        vecs = np.zeros([user_clicked_vec_shape[1]]+list(doc_input_shape))
        length = len(inputs)
        if length > user_clicked_vec_shape[1]:
            length = user_clicked_vec_shape[1]
        for i in range(-1,-1-length,-1):
            inp = inputs[i]
            if inp in self.doc_vec:
                vecs[i,:] = doc2title[inp]
        
        docs.append(vecs)
        
        docs = np.array(docs)
        
        for doc,title in doc2title.items():
            mydoc = doc
            candidate = title
            break
        
        pred = self.model.predict([docs,candidate])
        
        score_encoder = self.get_score_encoder()
        outputs = score_encoder.predict([np.stack([self.user_vec[myuser]]),np.stack([self.doc_vec[mydoc]])])
        
        sigm = 1/(1+np.exp(-outputs))
        
        print(pred)
        print(sigm)


class TestPipelineProduct(TestPipeline):
    def get_score_encoder(self):
        if self.score_encoder is not None:
            return self.score_encoder
        
        model = self.model
        input1 = keras.layers.Input(model.get_layer('user_encoder').output_shape[1:])
        input2 = keras.layers.Input(model.get_layer('doc_encoder').output_shape[1:])
        score = keras.layers.Dot(([1,1]))([input1,input2])
        self.score_encoder = keras.Model([input1,input2],score)
        return self.score_encoder
    
    
class TestPipelineBody(TestPipeline):
    def _test_doc_vec_gen(self):
        docs_path = self.config.pipeline_inputs[0]
        with utils.open(docs_path) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                doc = line[0]
                inputs = line[2]
                yield doc,inputs
            
    
    def get_doc_parser(self):
        doc_encoder = self.model.get_layer('doc_encoder')
        
        doc_input_shape = doc_encoder.layers[0].input_shape[1:]
        
        split_tokens = []
        with utils.open(self.config.doc_punc_index_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                if line[0] == '.' or line[0] == '?' or line[0] == '!':
                    split_tokens.append(int(line[1]))
                    
        
        body_parser = document.DocumentParser(
            document.parse_document(),
            document.clause(split_tokens),
            document.pad_docs(doc_input_shape[0], doc_input_shape[1])
        )
        
        return body_parser
        