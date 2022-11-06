import splade
'''
Utilising transformer base classes from SPLADE implementation: https://github.com/naver/splade/
'''

class sparseGradModel(splade.models.SiameseBase):
    def __init__(self, model_type_or_dir, model_type_or_dir_q=None, freeze_d_model=False, fp16=True):
        super().__init__(model_type_or_dir=model_type_or_dir,
                         output="MLM",
                         match="dot_product",
                         model_type_or_dir_q=model_type_or_dir_q,
                         freeze_d_model=freeze_d_model,
                         fp16=fp16)
        self.output_dim = self.transformer_rep.transformer.config.vocab_size  
        
    def encode(self, tokens, is_q):
        out = self.encode_(tokens, is_q)

