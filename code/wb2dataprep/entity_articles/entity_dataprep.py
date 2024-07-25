from wb0configs import configs
from wb0configs.helpers import store_file, load_file, store_xml, load_xml
from wb2dataprep.parsing.preprocess_parsing import PreprocessParser





class EntityPreprocessParser(PreprocessParser):

    def __init__(self,config):
        super().__init__(config)
        self.conflict_entity = dict()

    def pre_cleaning_section(self, sectioncontent, pageid, pagetitle):
        pass
        return sectioncontent

    def post_cleaning_section(self, sectioncontent, pageid, pagetitle):
        pass
        return sectioncontent



