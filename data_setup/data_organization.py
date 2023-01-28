"""DataProcessing.py: Cleans up data, handling missing values and imputations"""

__author__      = "Akira Nair"
__project__     = "Singh Lab - TCGA Project LUAD"

import glob
from itertools import count
import numpy as np
import shutil
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# patient_tab_data = MainDataset.readTSV("Data/DraftData/clinical data/nationwidechildrens.org_clinical_patient_luad.txt")
# sample_sheet_data = readTSV("NewData/gdc_sample_sheet.2022-03-31.tsv")

os.chdir("/users/anair27/data/anair27")
class DataByCases:
    global NEW_DATA_PATH
    global ORIG_DATA_PATH
    os.chdir("/users/anair27/data/anair27")
    NEW_DATA_PATH = "/users/anair27/data/anair27/data_by_cases"
    ORIG_DATA_PATH = "/users/anair27/data/anair27/data_original"
    PATIENT_TAB_PATH = "./data_by_cases/clinical_patient_luad.txt"
    SAMPLE_SHEET_PATH = "./data_by_cases/gdc_sample_sheet_luad.tsv"
    """ Reorganizes the directory to hold the downloaded data by cases """
    def __init__(self, patient_tab_path = PATIENT_TAB_PATH, sample_sheet_path = SAMPLE_SHEET_PATH):
        """Initializes the dataset"""
        patient_tab_data = self.readTSV(patient_tab_path)
        self.samples = self.readTSV(sample_sheet_path)
        # cases contains a list of all the case IDs
        self.cases = np.delete(patient_tab_data.loc[:, 'case_submitter_id'].to_numpy(), [0,1])
        # cncdata contains the clinical data for all patients
        # self.cncdata = patient_tab_data.loc[:,['bcr_patient_barcode', 'death_days_to']].iloc[2:,:]
        # create the outcome variable, survival, for each patient and append it to cncdata
        # self.create_outcome_variable()
        # if new_data_path does not exist, generate the case folders        
        
    def reorganize(self, new_data_path = NEW_DATA_PATH):
        if not os.path.isdir(new_data_path):
            print("Generating new data path.")
            os.mkdir(NEW_DATA_PATH)
        else:
            print("New data path already exists.")
        print("Adding case folders.")
        self.create_case_folders()
        #print("Moving data into new data path.")
        #self.move_data()
            

    def create_outcome_variable(self):
        """ creates the survival outcome variable """
        death = self.cncdata.loc[:,'death_days_to'].to_numpy()
        outcome = []
        for d in death:
            if d == '[Not Applicable]' or d == '[Not Available]':
                outcome.append(1)
            elif int(d) > 400:
                outcome.append(1)
            else:
                outcome.append(-1)
        self.cncdata['survival_outcome'] = outcome

    def create_case_folders(self):
        count = 0
        for case in self.cases:
            path = NEW_DATA_PATH + "/"+ case
            print("Creating path", path)
            try:
                os.mkdir(path)
                self.create_data_folders(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                count += 1
                print(" Successfully created directory %s " % path)
        print(count)

    def create_data_folders(self, path):
        count = 0
        dna_methylation_subpath = path + "/dna_methylation"
        cnv_subpath = path + "/cnv"
        gene_exp_subpath = path + "/gene_expression"
        images_subpath = path + "/images"
        other_subpath = path + "/other"
        try:
            os.mkdir(dna_methylation_subpath)
            os.mkdir(cnv_subpath)
            os.mkdir(gene_exp_subpath)
            os.mkdir(images_subpath)
            os.mkdir(other_subpath)
        except OSError:
            print("Creation of a subdirectory failed")
        else:
            count += 1
            print("Successfully created subdirectories")
        print(count)
    
    def move_data(self):
        count = 0
        for index,row in self.samples.iterrows():
            assoc_case = row['Case ID'].split(",",1)[0]
            assoc_type = row['Data Type']
            assoc_file = row['File ID']
            assoc_file_name = row['File Name']
            source = ORIG_DATA_PATH + "/" + assoc_file + "/" + assoc_file_name
            dest = NEW_DATA_PATH + "/" + assoc_case + self.type_extension(assoc_type)+"/"
            print("Moving", source, "to", dest)
            try:
                shutil.move(source, dest)
                count +=1
            except:
                print("Error in moving data from original data to new data")
                print("Filename: " + assoc_case + " - " + assoc_file_name)

                # raise FileNotFoundError("Error in moving file ")
            # source_dir = "/data/singhlab/utra2020/cnv_meth27_data/original_data/" + row[0]
            # dest_dir = "/data/singhlab/utra2020/cnv_meth27_data/data_by_cases/" + row[1][:12]
            # shut.move(source_dir, dest_dir)
        print("Files transferred:", count)

    def type_extension(self, assoc_type):
        if assoc_type == "Gene Expression Quantification":
            return "/gene_expression"
        elif assoc_type == "Gene Level Copy Number":
            return "/cnv"
        elif assoc_type == "Slide Image":
            return "/images"
        elif assoc_type == "Methylation Beta Value":
            return "/dna_methylation"
        else:
            return "/other"
    
    def readTSV(self, address):
        return pd.read_csv(address, sep = "\t")

class MainDataset:
    """ A structural framework to represent the data"""
    def __init__(self, cases) -> None:
        self.cases_list = cases
        self.case_set = []
        for caseID in self.cases_list:
            self.case_set.append(Case(caseID))
    def create_partition(self):
        train, validation = train_test_split(self, test_size=0.50, random_state = 1)
        pass
    

class Case:
    def __init__(self, case_ID) -> None:
        self.id = str(case_ID)
        self.import_case()
        self.has_data()

    def import_case(self) -> None:
        self.exp_data = self.import_gene_expression()
        self.cnv_data = self.import_cnv()
        self.image_data = self.import_images()
        #self.methylation_data = self.import_methylation()
    
    def import_gene_expression(self):
        pass

    def import_cnv(self):
        pass

    def import_images(self):
        pass

    def import_methylation(self):
        file_add = NEW_DATA_PATH+"/"+self.id+"/dna_methylation/*.txt"
        file_paths = glob.glob(file_add)
        if len(file_paths) == 0:
            return None
        else:
            return pd.read_csv(glob.glob(dir)[0], sep = '\t')    

    def has_data(self):

        gene_expression_dir = NEW_DATA_PATH+"/"+self.id+"/gene_expression/"
        cnv_dir = NEW_DATA_PATH+"/"+self.id+"/cnv/"
        images_dir = NEW_DATA_PATH+"/"+self.id+"/images/"
        dna_methylation_dir = NEW_DATA_PATH+"/"+self.id+"/dna_methylation/"
        
        def empty_dir(dir):
            return True if len(os.listdir(dir))==0 else False
        
        if empty_dir(gene_expression_dir) or empty_dir(cnv_dir)\
            or empty_dir(images_dir) or empty_dir(dna_methylation_dir):
            self.is_complete = False
        else:
            self.is_complete = True

    def __str__(self) -> str:
        return self.id

def main():
    data_by_cases = DataByCases()
    data_by_cases.reorganize()
    full_dataset = MainDataset(data_by_cases.cases)
    

# c1 = Case("442304")
# print(c1, "status:", c1.import_methylation()[1])
# def cases_to_data_mapping(cases):
#     """Creates a list of tuples containing case ID, methylation file ID, CNV file id, RNA seq file ID"""
#     cases_to_data = []
#     for case in cases:
#         if case_complete(case):
#             methylation_file = get_methylation_file(case)
#             cnv_file = get_cnv_file(case)
#             RNAseq_file = get_RNAseq_file(case)
#             cases_to_data.append((case, methylation_file, cnv_file,RNAseq_file))
        



    # def create_methylation_fileID(self, methylation_manifest_add):
    #     methylationFileIDs = getFileUUIDs(methylation_manifest_add)
    #     patient_list = self.data['bcr_patient_uuid']
    #     for file in methylationFileIDs:
    #         if fileToCase(file) in patient_list:
    #             print("patient has methylation file!")
    #         else:
    #             print("nope.")