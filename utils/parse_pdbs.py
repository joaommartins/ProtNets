# Copyright (c) 2017 Jes Frellsen and Wouter Boomsma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import os
import sys
import glob
import Bio.PDB
import tempfile
import subprocess
import simtk
import simtk.openmm
import simtk.openmm.app
import pdbfixer
import Bio.PDB.Polypeptide
import Bio.SeqIO
import StringIO

if not 'TRN' in pdbfixer.pdbfixer.substitutions:
    pdbfixer.pdbfixer.substitutions['TRN'] = 'TRP'

class IncompleteSeqResError(Exception):
    pass

class IncompletePDBError(Exception):
    pass

class ChainBreakError(Exception):
    pass

class OpenMMParseError1(Exception):
    pass

class OpenMMParseError2(Exception):
    pass

class OpenMMException(Exception):
    pass

class PDBNotFoundError(Exception):
    pass

class ReduceError(Exception):
    pass

class PDBFixerResIdentifiabilityIssue(Exception):
    pass


def parse_pdb(pdb_filehandle, pdb_id, reduce_executable, dssp_executable, chain_id_filter=None, max_N_terminal_residues=10000, use_pdb_fixer=True, allow_chain_breaks=False, allow_incomplete_pdb=False, verbose=False):

    # fixer = pdbfixer.PDBFixer(pdbfile=pdb_filehandle)

    # chains_to_be_removed = []
    # for chain in fixer.topology.chains():
    #     if chain.id is not chain_id:
    #         chains_to_be_removed.append(chain.id)
    # fixer.removeChains(chainIds=chains_to_be_removed)

    # fixer.findMissingResidues()

    class ChainAndNonHetSelector(Bio.PDB.Select):
        """ Only accept the specified chains when saving. """
        def __init__(self, chain_letters):
            self.chain_letters = chain_letters

        def accept_residue(self, residue):
            print residue.get_resname(), residue.id[0] == " ", residue.get_resname() in pdbfixer.pdbfixer.substitutions
            return residue.id[0] == " " or (residue.get_resname() in pdbfixer.pdbfixer.substitutions)
            # return residue.get_resname() != "HOH"
        #     return 1 if residue.id[0] == " " else 0

        def accept_model(self, model):
            return model.id == 0
        
        def accept_chain(self, chain):
            if self.chain_letters is None:
                return True
            return (chain.get_id() == ' ' or chain.get_id() in self.chain_letters)

        def accept_atom(self, atom):
            # print "atom?: ", atom, atom.get_parent().id, atom.is_disordered(), (not atom.is_disordered() or atom.get_altloc() == 'A' or atom.get_altloc() == '1')
            return not atom.is_disordered() or atom.get_altloc() == 'A' or atom.get_altloc() == '1'
    # class NonHetSelect(Bio.PDB.Select):
    #     def accept_residue(self, residue):
    #         return 1 if residue.id[0] == " " else 0

    pdb_content = pdb_filehandle.read()

    pdb_filehandle = StringIO.StringIO(pdb_content)

    seqres_seqs = {}
    for record in Bio.SeqIO.PdbIO.PdbSeqresIterator(pdb_filehandle):
        seqres_seqs[record.annotations["chain"]] = str(record.seq)
    # print seqres_seqs
    
    pdb_filehandle = StringIO.StringIO(pdb_content)
    pdb_parser = Bio.PDB.PDBParser(PERMISSIVE=0)
    structure = pdb_parser.get_structure(pdb_id, pdb_filehandle)
    io = Bio.PDB.PDBIO()
    io.set_structure(structure)

    first_model = structure.get_list()[0]
    ppb = Bio.PDB.PPBuilder()
    for i, chain in enumerate(first_model):
        pps = ppb.build_peptides(chain)

        seq = ""
        for atom in chain.get_atoms():
            if atom.id == 'CA':
                try:
                    aa = Bio.PDB.Polypeptide.three_to_one(atom.get_parent().get_resname())
                except:
                    aa = 'X'
                seq += aa
        # seq = "".join([Bio.PDB.Polypeptide.three_to_one(atom.get_parent().get_resname()) for atom in chain.get_atoms() if atom.id == 'CA'])

        chain_id = chain.id


        if not allow_chain_breaks:
            number_of_pps = len(list(pps))
            if not (number_of_pps == 1 and len(pps[0]) == len(seq)) :
                if verbose:
                    print "chain: ", i
                    for pp in pps:
                        print pp.get_sequence()
                raise ChainBreakError


        if not allow_incomplete_pdb:
            if chain_id not in seqres_seqs:
                raise IncompleteSeqResError() 
            seqres_seq = seqres_seqs[chain_id]
            if (len(seq) != len(seqres_seq)):
                raise IncompletePDBError({'message':'\n'+seq+'\n'+seqres_seq})
            
        # print pps

    # for record in Bio.SeqIO.parse(pdb_filehandle, "pdb-seqres"):
    #     print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
    #     print(record.dbxrefs)
    
    first_residue_index = structure[0].get_list()[0].get_list()[0].get_id()[1]
    
    # fixer.findNonstandardResidues()
    # fixer.replaceNonstandardResidues()

    # fixer.findMissingAtoms()
    # fixer.addMissingAtoms()

    # fixer.removeHeterogens(False)

    # fixer.addMissingHydrogens(7.0)
    with tempfile.NamedTemporaryFile(delete=False) as temp1:

        print "\t", temp1.name
        
        # simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, temp1)
        # io.save(temp1, ChainAndNonHetSelector(chain_id))
        io.save(temp1, ChainAndNonHetSelector(chain_id_filter))        
        temp1.flush()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp2:
            # simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, temp)
            # temp.flush()

            # reduce_command_line = "%s -BUILD -Quiet -DB %s %s > %s" % (reduce_executable,
            #                                                            os.path.join(os.path.dirname(reduce_executable), 'reduce_wwPDB_het_dict.txt'),
            #                                                            temp1.name, temp2.name)
            # os.system(reduce_command_line)
            print "\t", temp2.name

            command = [reduce_executable, '-BUILD',
                                           '-DB', os.path.join(os.path.dirname(reduce_executable), 'reduce_wwPDB_het_dict.txt'),
                                           '-Quiet',
                                           # '-Nterm'+str(max_N_terminal_residues),
                                           '-Nterm'+str(first_residue_index),                       
                                           temp1.name]
            print " ".join(command)
            error_code = subprocess.Popen(command,
                                          stdout=temp2).wait()
            # if error_code < 0:
            #     raise ReduceError("Reduce program returned a negative error code value.")
                                 # stdout=output)
            # p.communicate(input=pdb_fixer_output.getvalue())
            temp2.flush()

            # Use PDBFixer to fix common PDB errors
            fixer = pdbfixer.PDBFixer(temp2.name)

            if use_pdb_fixer:
                # print "Running pdb fixer"
                # chains_to_be_removed = []
                # for chain in fixer.topology.chains():
                #     if chain.id is not chain_id:
                #         chains_to_be_removed.append(chain.id)
                # fixer.removeChains(chainIds=chains_to_be_removed)

                fixer.findMissingResidues()

                fixer.findNonstandardResidues()
                fixer.replaceNonstandardResidues()
                
                # Remove waters and other non-protein atoms
                # fixer.removeHeterogens(False)

                fixer.findMissingAtoms()

                try:
                    fixer.addMissingAtoms()
                    fixer.addMissingHydrogens(7.0)
                except Exception as e:
                    raise OpenMMException(e.message)
                
                with tempfile.NamedTemporaryFile(delete=False) as temp3:

                    print "\t", temp3.name
                    # We would have liked to use keepIds=True here, but it does not preserve insertion codes,
                    # so we instead set the IDs manually
                    simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, temp3, keepIds=False)
                    temp3.flush()

                    pdb_parser = Bio.PDB.PDBParser(PERMISSIVE=1)
                    structure_before = pdb_parser.get_structure(temp2.name, temp2.name)
                    structure_after = pdb_parser.get_structure(temp3.name, temp3.name)

                    # for residue in pdb.getTopology().chains()[0].residues():

                    # PDBfixer does not preserve insertion codes. We therefore do it manually here
                    # residues_before = structure_before[0].get_list()[0].get_list()
                    residues_before = []
                    for chain in structure_before[0]:
                        residues_before.append(chain.get_list())
                    # residues_after = structure_after[0].get_list()[0].get_list()
                    residues_after = []
                    for chain in structure_after[0]:
                        residues_after.append(chain.get_list())
                    # print residues_before
                    # print residues_after
                    # print (len(residues_before), len(residues_after))
                    for i,chain in enumerate(structure_before[0]):
                        structure_after[0].get_list()[i].id = structure_before[0].get_list()[i].id
                        # print structure_before[0].get_list()[i], structure_after[0].get_list()[i]
                        if len(residues_before[i]) != len(residues_after[i]):
                            raise PDBFixerResIdentifiabilityIssue()
                        for res1, res2 in zip(residues_before[i], residues_after[i]):
                            assert(res1.get_resname().strip() == res2.get_resname().strip() or pdbfixer.pdbfixer.substitutions[res1.get_resname()].strip() == res2.get_resname().strip())
                            res2.id = res1.id
                            # print res2.id

                    io = Bio.PDB.PDBIO()
                    io.set_structure(structure_after)
                    with tempfile.NamedTemporaryFile(delete=False) as temp4:
                        print "\t", temp4.name
                        
                        io.save(temp4)        
                        temp4.flush()
                        
                        # Read in PDB file
                        try:
                            pdb = simtk.openmm.app.PDBFile(temp4.name)
                        except:
                            raise OpenMMParseError1

                        structure = structure_after

            else:

                # Read in PDB file
                pdb = simtk.openmm.app.PDBFile(temp2.name)
                
                pdb_parser = Bio.PDB.PDBParser(PERMISSIVE=0)
                structure = pdb_parser.get_structure(temp2.name, temp2.name)

                # Attempt to extract DSSP
                first_model = structure.get_list()[0]
                dssp = Bio.PDB.DSSP(first_model, temp2.name, dssp=dssp_executable)
                for i, chain in enumerate(first_model):
                    pps = ppb.build_peptides(chain)
                    ss = np.array([dssp2i(res.xtra["SS_DSSP"]) for res in chain], dtype=np.int8)        

                
                # structure_new = pdb_parser.get_structure(temp3.name, temp3.name)
                # structure_old = pdb_parser.get_structure(temp2.name, temp2.name)

                # print temp2.name, temp3.name
                # # Check that the chain is unbroken
                # ppb = Bio.PDB.PPBuilder()
                # coordinates = []
                # for pp1, pp2 in zip(ppb.build_peptides(structure_new),
                #                     ppb.build_peptides(structure_old)):
                #     for i in range(len(pp1)):
                #         res1, res2 = pp1[i], pp2[i]
                #         # print res1['N'].coord, res2['N'].coord
                #         coordinates.append([res1['N'].coord, res2['N'].coord])
                #         coordinates.append([res1['CA'].coord, res2['CA'].coord])
                #         coordinates.append([res1['C'].coord, res2['C'].coord])
                # coordinates = np.array(coordinates)
                # print np.mean((coordinates[:,1,:] - coordinates[:,0,:])**2)

    # if not allow_chain_breaks:
    #     # Check that the chain is unbroken
    #     ppb = Bio.PDB.PPBuilder()
    #     CA_seq = "".join([Bio.PDB.Polypeptide.three_to_one(atom.get_parent().get_resname()) for atom in structure[0].get_atoms() if atom.id == 'CA'])
    #     pps = ppb.build_peptides(structure)
    #     number_of_pps = len(list(pps))
    #     if not (number_of_pps == 1 and len(pps[0]) == len(CA_seq)) :
    #         print len(pps), len(pps[0]), len(CA_seq)
    #         print CA_seq
    #         for pp in pps:
    #             print pp.get_sequence()
    #         raise Exception("Chain breaks detected")

    # Extract positions
    positions = pdb.getPositions()

    # Create forcefield in order to extract charges
    forcefield = simtk.openmm.app.ForceField('amber99sb.xml', 'tip3p.xml')

    # Create system to couple topology with forcefield
    # system = forcefield.createSystem(pdb.getTopology(), ignoreExternalBonds=True)
    try:
        system = forcefield.createSystem(pdb.getTopology())
    except ValueError as e:
        print e
        raise OpenMMParseError2

    return structure


if __name__ == '__main__':

    from argparse import ArgumentParser
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = ArgumentParser()
    parser.add_argument("--pdb-input-dir", dest="pdb_input_dir",
                        help="Location of pdbs")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Where to dump features")
    parser.add_argument("--reduce-executable", dest="reduce_executable",
                        help="Location of reduce executable")
    parser.add_argument("--dssp-executable", dest="dssp_executable",
                        help="Location of dssp executable")
    parser.add_argument("--allow-chain-breaks", dest="allow_chain_breaks",
                        type=str2bool, nargs='?', const=True, default="False",
                        help="Whether to allow chain breaks in PDB")
    parser.add_argument("--allow-incomplete-pdb", dest="allow_incomplete_pdb",
                        type=str2bool, nargs='?', const=True, default="False",
                        help="Whether to allow mismatch between PDB and seqres record")
    parser.add_argument("--use-pdb-fixer", dest="use_pdb_fixer",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Whether to use PDB fixer")
    parser.add_argument("--verbose", dest="verbose",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Output additional information")

    options = parser.parse_args()

    pdb_filenames = glob.glob(options.pdb_input_dir + "/*")

    if not os.path.exists(options.output_dir):
        os.mkdir(options.output_dir)
    
    for pdb_filename in pdb_filenames:

        handle = open(pdb_filename)

        pdb_id = os.path.basename(pdb_filename).replace(".pdb", "").split('_')[0]
        # chain_id = pdb_id[-1]
        
        print pdb_filename
        try:
            structure = parse_pdb(handle, pdb_id, options.reduce_executable, options.dssp_executable, use_pdb_fixer=options.use_pdb_fixer, allow_chain_breaks=options.allow_chain_breaks, allow_incomplete_pdb=options.allow_incomplete_pdb, verbose=options.verbose)
        except IncompletePDBError as e:
            print e.message.values()[0]
            raise
        # except Exception as e:
        #     print "%s Failed: %s" % (pdb_filename, e)
        #     continue

        io = Bio.PDB.PDBIO()
        io.set_structure(structure)
        io.save(os.path.join(options.output_dir, pdb_id + ".pdb"))
        
