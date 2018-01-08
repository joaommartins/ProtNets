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

import glob
import os
import warnings

import Bio.PDB
import Bio.PDB.DSSP
# import h5py
import Bio.PDB.Vector
import numpy as np
import simtk
import simtk.openmm
import simtk.openmm.app
import simtk.unit

warnings.simplefilter(action = "ignore", category = FutureWarning)

from utils import grid

from utils.parse_pdbs import parse_pdb


# DSSP to SS-index(H, E, C)
# H: HGI
# E: EB
# C: STC
def dssp2i(ss):
    '''Convert 8-class DSSP score into three numeric classes'''
    if ss in ("H", "G", "I"):
        return 0
    elif ss in ("E","B"):
        return 1
    elif ss in ("S", "T", "C", " "):
        return 2
    else:
        # Used for unobserved SS
        return 3


def extract_mass_charge(pdb_filename, dssp_executable):
    '''Extract protein-level features from pdb file'''
    
    pdb_id = os.path.basename(pdb_filename).split('.')[0]

    # Read in PDB file
    pdb = simtk.openmm.app.PDBFile(pdb_filename)

    # Also parse through Bio.PDB - to extract DSSP secondary structure info
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_id, pdb_filename)

    # dssp = Bio.PDB.DSSP(structure[0], pdb_filename, dssp=dssp_executable)
    
    first_model = structure.get_list()[0]
    ppb = Bio.PDB.PPBuilder()
    sequence = []
    aa_one_hot = []
    ss_one_hot = []
    chain_ids = []
    for i, chain in enumerate(first_model):
        pps = ppb.build_peptides(chain)

        chain_ids.append(chain.id)
        # if not allow_chain_breaks:
        #     number_of_pps = len(list(pps))
        #     if not (number_of_pps == 1 and len(pps[0]) == len(seq)) :
        #         # for pp in pps:
        #         #     print pp.get_sequence()
        #         raise ChainBreakError

        sequence_chain = []
        for res in chain.get_residues():
            sequence_chain.append(res.resname.strip())
            
        # for atom in chain.get_atoms():
        #     if atom.id == 'CA':
        #         try:
        #             aa = Bio.PDB.Polypeptide.three_to_one(atom.get_parent().get_resname())
        #         except:
        #             aa = 'X'
        #         sequence_chain += aa
                
        sequence.append(sequence_chain)

        
        aa_indices = []
        for aa in sequence_chain:
            try:
                aa_index = Bio.PDB.Polypeptide.three_to_index(aa)
            except:
                aa_index = 20
            aa_indices.append(aa_index)
        aa_one_hot_chain = np.zeros((len(aa_indices), 21))
        aa_one_hot_chain[np.arange(len(aa_indices)), aa_indices] = 1
        
        # Calculate secondary structure
        ss = []
        for res in chain:
            try:
                ss.append(dssp2i(res.xtra["SS_DSSP"]))
            except:
                ss.append(3)
        ss = np.array(ss, dtype=np.int8)
        ss_one_hot_chain = np.zeros((len(ss), 4))
        ss_one_hot_chain[np.arange(len(ss)), ss] = 1

        aa_one_hot.append(aa_one_hot_chain)
        ss_one_hot.append(ss_one_hot_chain)
        
    chain_boundary_indices = np.cumsum([0]+[len(entry) for entry in aa_one_hot])
    aa_one_hot = np.concatenate(aa_one_hot)
    ss_one_hot = np.concatenate(ss_one_hot)

    # print chain_boundary_indices
    # ppb = Bio.PDB.PPBuilder()
    # pps = list(ppb.build_peptides(structure))
    # if len(pps) != 1:
    #     raise Exception("Chain breaks detected")
    # sequence = pps[0].get_sequence()
    # aa_indices = [Bio.PDB.Polypeptide.one_to_index(aa) for aa in sequence]
    # aa_one_hot = np.zeros((len(aa_indices), 20))
    # aa_one_hot[np.arange(len(aa_indices)), aa_indices] = 1
    
    # # try:
    # #     print pdb_filename
    # # dssp = Bio.PDB.DSSP(model=structure[0], pdb_file=pdb_filename, dssp=dssp_executable)
    # # except Exception as e:
    # #     print e
    # #     dssp = Bio.PDB.DSSP(model=structure[0], in_file=pdb_filename, dssp=dssp_executable)

    # # Calculate secondary structure
    # dssp = Bio.PDB.DSSP(structure[0], pdb_filename, dssp=dssp_executable)
    # ss = np.array([dssp2i(res.xtra["SS_DSSP"]) for res in structure[0].get_list()[0]], dtype=np.int8)
    # ss_one_hot = np.zeros((len(ss), 3))
    # ss_one_hot[np.arange(len(ss)), ss] = 1
    
    # chain_id = pdb_id[4]

    # Use PDBFixer to fix common PDB errors
    # fixer = pdbfixer.PDBFixer(pdb_filename)

    # # Remove chains (there should only be one, this might not be necessary)
    # chains_to_be_removed = []
    # for chain in fixer.topology.chains():
    #     if chain.id is not chain_id:
    #         chains_to_be_removed.append(chain.id)
    # fixer.removeChains(chainIds=chains_to_be_removed)

    # fixer.findMissingResidues()
    # fixer.findNonstandardResidues()
    # fixer.replaceNonstandardResidues()

    # # fixer.findMissingAtoms()
    # # fixer.addMissingAtoms()

    # # Remove waters and other non-protein atoms
    # fixer.removeHeterogens(False)

    # pdb_fixer_output = StringIO.StringIO()
    # simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, pdb_fixer_output)

    # class NonHetSelect(Bio.PDB.Select):
    #     def accept_residue(self, residue):
    #         return 1 if residue.id[0] == " " else 0

    # pdb_parser = Bio.PDB.PDBParser(PERMISSIVE=0)
    # structure = pdb_parser.get_structure(pdb_filename, pdb_filename)
    # io = Bio.PDB.PDBIO()
    # io.set_structure(structure)

    # # Add hydrogens at neutral pH
    # fixer.addMissingHydrogens(7.0)
    # Write new PDB to temporary file
    # with tempfile.NamedTemporaryFile() as temp1:

    #     io.save(temp1, NonHetSelect())
    #     temp1.flush()
            
    #     with tempfile.NamedTemporaryFile() as temp2:

    #         # Run reduce
    #         error_code = subprocess.Popen([reduce_executable, '-BUILD',
    #                                        '-DB', os.path.join(os.path.dirname(reduce_executable), 'reduce_wwPDB_het_dict.txt'),
    #                                        '-Quiet',
    #                                        temp1.name],
    #                                       stdout=temp2).wait()
    #         if error_code < 0:
    #             raise Exception("Reduce program returned a negative error code value.")

    #         temp2.flush()
            
    #         # print reduce_command_line
    #         # os.system(reduce_command_line)
    #         # output_name = pdb_id+'.pdb'
    #         # output = open(output_name, 'w')
    #         # reduce_command_line = "%s -BUILD -Quiet -DB %s %s > %s" % (reduce_executable,
    #         #                                                            os.path.join(os.path.dirname(reduce_executable), 'reduce_wwPDB_het_dict.txt'),
    #         #                                                            pdb_filename, output_name)
    #         # print reduce_command_line
    #         # os.system(reduce_command_line)
    #         # p = subprocess.Popen([reduce_executable, '-BUILD',
    #         #                       '-DB', os.path.join(os.path.dirname(reduce_executable), 'reduce_wwPDB_het_dict.txt'),
    #         #                       '-Quiet',
    #         #                       '-'],
    #         #                      stdin=subprocess.PIPE,
    #         #                      # stdout=temp)
    #         #                      stdout=output)
    #         # p.communicate(input=pdb_io_output.getvalue())
    #         # output.close()
    #         # temp.flush()
    #         # print output.getvalue()

    #         # with tempfile.NamedTemporaryFile() as temp2:

    #         #     output_name2 = pdb_id+'_pdbfixed.pdb'
    #         #     output2 = open(output_name2, 'w')
    #         #     fixer = pdbfixer.PDBFixer(output_name)
    #         #     fixer.removeHeterogens(False)
    #         #     simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, output2)
    #         #     output2.close()


    #         # # Write file
    #         # simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, temp)
    #         # temp.flush()

    #         # # Read in PDB file
    #         # pdb = simtk.openmm.app.PDBFile(temp2.name)

    #         # # Also parse through Bio.PDB - to extract DSSP secondary structure info
    #         # pdb_parser = Bio.PDB.PDBParser()
    #         # structure = pdb_parser.get_structure(temp2.name, temp2.name)
    #         # try:
    #         #     dssp = Bio.PDB.DSSP(model=structure[0], pdb_file=temp2.name, dssp=dssp_executable)
    #         # except:
    #         #     dssp = Bio.PDB.DSSP(model=structure[0], in_file=temp2.name, dssp=dssp_executable)

    #         # ss = np.array([dssp2i(res.xtra["SS_DSSP"]) for res in structure[0].get_list()[0]], dtype=np.int8)
        
    # Extract positions
    positions = pdb.getPositions()

    # Create forcefield in order to extract charges
    forcefield = simtk.openmm.app.ForceField('amber99sb.xml', 'tip3p.xml')

    # Add hydrogens if necessary
    modeller = simtk.openmm.app.Modeller(pdb.getTopology(), pdb.getPositions())
    
    # Create system to couple topology with forcefield
    system = forcefield.createSystem(modeller.getTopology())

    # Find nonbonded force (contains charges)
    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, simtk.openmm.openmm.NonbondedForce):
            nonbonded_force = force

    # Create structured array for features
    features = np.empty(shape=(len(positions), 1), dtype=[('mass',np.float32),
                                                          ('charge',np.float32),
                                                          ('name','a5'),
                                                          ('res_index', int),
                                                          ('x',np.float32), ('y',np.float32), ('z',np.float32)])
    # Iterate over chain,residue,atoms and extract features
    for i,chain in enumerate(pdb.getTopology().chains()):
        chain_start_index = chain_boundary_indices[i]
        for j,residue in enumerate(chain.residues()):
            # print residue.id, residue.name
            for atom in residue.atoms():
                index = atom.index
                position = list(positions[index].value_in_unit(simtk.unit.angstrom))
                mass = atom.element.mass.value_in_unit(simtk.unit.dalton)
                charge = nonbonded_force.getParticleParameters(index)[0].value_in_unit(simtk.unit.elementary_charge)
                features[index] = tuple([mass, charge, atom.name, residue.index]+position)

                residue_index_local = residue.index - chain_start_index
                # print i, j, sequence, residue_index_local, residue.index, chain_start_index
                # print ">%s<" % residue.name, ">%s<" % sequence[i][residue_index_local]
                # print residue.name, sequence[i][residue_index_local]
                assert(residue.name == sequence[i][residue_index_local])

    # Convert relevant entries to standard numpy arrays
    masses_array = features[['mass']].view(np.float32)
    charges_array = features[['charge']].view(np.float32)

    # check that ss input matches number of residues
    # assert(len(ss) == len(np.unique(features[['res_index']].view(int))))
    # sequence_distances = features[['res_index']].view(int)
    res_index_array = features[['res_index']].view(int)

    return pdb_id, features, masses_array, charges_array, aa_one_hot, ss_one_hot, res_index_array, chain_boundary_indices, chain_ids


def embed_in_grid(features, pdb_id, output_dir,
                  max_radius,
                  n_features,
                  bins_per_angstrom,
                  coordinate_system,
                  z_direction,
                  include_center):
    '''Embed masses and charge information in a spherical grid - specific for each residue
       For space-reasons, only the indices into these grids are stored, and a selector 
       specifying which atoms are relevant (i.e. within range) for the current residue.
    '''

    # Extract coordinates as normal numpy array
    position_array = features[['x','y','z']].view(np.float32)

    selector_list = []
    indices_list = []
    for residue_index in np.unique(features[['res_index']].view(int)):

        res_indices = features[['res_index']].view(int)
        
        # Extract origin
        if (np.logical_and(res_indices == residue_index, features[['name']].view('a5') == "CA").any() and
            np.logical_and(res_indices == residue_index, features[['name']].view('a5') == "CA").any() and
            np.logical_and(res_indices == residue_index, features[['name']].view('a5') == "CA").any()):
            CA_feature = features[np.argmax(np.logical_and(res_indices == residue_index, features[['name']].view('a5') == "CA"))]
            N_feature = features[np.argmax(np.logical_and(res_indices == residue_index, features[['name']].view('a5') == "N"))]
            C_feature = features[np.argmax(np.logical_and(res_indices == residue_index, features[['name']].view('a5') == "C"))]
        else:
            # Store None to maintain indices
            indices_list.append(None)
            selector_list.append(None)
            continue

        pos_CA = CA_feature[['x','y','z']].view(np.float32)
        pos_N = N_feature[['x','y','z']].view(np.float32)
        pos_C = C_feature[['x','y','z']].view(np.float32)

        # Define local coordinate system
        rot_matrix = grid.define_coordinate_system(pos_N, pos_CA, pos_C, z_direction)
        # e1 = (pos_C-pos_N)
        # e1 /= np.linalg.norm(e1)
        # e2 = (pos_CA-pos_N)+(pos_CA-pos_C)
        # e2 /= np.linalg.norm(e2)
        # # CB_entries = np.logical_and(features[['res_index']].view(int) == residue_index, features[['name']].view('a5') == "CB")
        # # if CB_entries.any():
        # #     pos_CB = features[np.argmax(CB_entries)][['x','y','z']].view(np.float32)
        # # else:
        # #     # Define CB positions by rotating N atoms around CA-C axis 120 degr
        # #     pos_N_res = pos_N - pos_CA
        # #     axis = pos_C - pos_CA
        # #     dir(Bio.PDB.Vector)
        # #     pos_CB = np.dot(Bio.PDB.rotaxis((120./180.)*np.pi, Bio.PDB.Vector(axis)), pos_N_res)
        # # e2 = (pos_CB-pos_CA)
        # # e2 /= np.linalg.norm(e2)
        # e3 = np.cross(e1, e2)

        # # N-C and e2 are not perfectly perpendical to one another. We adjust e2.
        # e2 = np.cross(e1, e3)

        # # Rotation matrix to apply to coordinates
        # rot_matrix = np.array([e1,e2,e3])
     
        # # Calculate distance along sequence
        # sequence_distances = sequence_distances - origin_CA_feature[['res_index']].view(int)

        # Calculate spherical coordinates
        xyz = position_array - pos_CA

        # print "xyz: ", xyz[4]

        # Rotate to the local reference
        xyz = np.dot(rot_matrix, xyz.T).T

        if coordinate_system == grid.CoordinateSystem.spherical:

            # Convert to spherical coordinates
            r, theta, phi = grid.cartesian_to_spherical_coordinates(xyz)

            # print features[0]
            # print xyz[0], r[0], theta[0], phi[0]
            # print features[1]
            # print xyz[1], r[1], theta[1], phi[1]
            # print features[2]
            # print xyz[2], r[2], theta[2], phi[2]
            # print features[4]
            # print xyz[4], r[4], theta[4], phi[4]

            # sys.exit()
            # xy = xyz[:,0]**2 + xyz[:,1]**2
            # r = np.sqrt(xy + xyz[:,2]**2)
            # theta = np.arctan2(np.sqrt(xy), xyz[:,2])  # polar angle - inclination from z-axis
            # phi =  np.arctan2(xyz[:,1], xyz[:,0])

            # Create grid
            grid_matrix = grid.create_spherical_grid(max_radius=max_radius, n_features=n_features, bins_per_angstrom=bins_per_angstrom)

            # Bin each dimension independently
            r_bin, theta_bin, phi_bin = grid.discretize_into_spherical_bins(r, theta, phi, max_radius,
                                                                            grid_matrix.shape[0],
                                                                            grid_matrix.shape[1],
                                                                            grid_matrix.shape[2])
            # r_boundaries = np.linspace(0, 10, grid_matrix.shape[0], endpoint=False)
            # r_boundaries += (r_boundaries[1]-r_boundaries[0])
            # theta_boundaries = np.linspace(0, np.pi, grid_matrix.shape[1], endpoint=False)
            # theta_boundaries += (theta_boundaries[1]-theta_boundaries[0])
            # phi_boundaries = np.linspace(-np.pi, np.pi, grid_matrix.shape[2], endpoint=False)
            # phi_boundaries += (phi_boundaries[1]-phi_boundaries[0])        
            # r_bin = np.digitize(r, r_boundaries)
            # theta_bin = np.digitize(theta, theta_boundaries)
            # phi_bin = np.digitize(phi, phi_boundaries)

            if np.count_nonzero(r_bin==0) > 1:
                print "Weird"
                print np.where(r_bin==0)

            # Merge bin indices into one array
            indices = np.vstack((r_bin, theta_bin, phi_bin)).transpose()

            assert(not np.any(theta_bin >= grid_matrix.shape[1]))

            if np.any(phi_bin >= grid_matrix.shape[2]):
                print np.where(phi_bin >= grid_matrix.shape[2])[0]
                print zip(phi, phi_bin)[np.where(phi_bin >= grid_matrix.shape[2])[0][0]]
                # print zip(phi, phi_bin)

            assert(not np.any(phi_bin >= grid_matrix.shape[2]))

        elif coordinate_system == grid.CoordinateSystem.cubed_sphere:

            # Convert to coordinates on the cubed sphere
            patch, r, xi, eta = grid.cartesian_to_cubed_sphere_vectorized(xyz[:, 0], xyz[:, 1], xyz[:, 2])


            # Create grid
            grid_matrix = grid.create_cubed_sphere_grid(max_radius=max_radius, n_features=n_features, bins_per_angstrom=bins_per_angstrom)

            # Bin each dimension independently
            patch_bin, r_bin, xi_bin, eta_bin = grid.discretize_into_cubed_sphere_bins(patch, r, xi, eta,
                                                                                       max_radius,
                                                                                       grid_matrix.shape[1],
                                                                                       grid_matrix.shape[2],
                                                                                       grid_matrix.shape[3])

            if np.count_nonzero(r_bin==0) > 1:
                print "Weird"
                print np.where(r_bin==0)

            # Merge bin indices into one array
            indices = np.vstack((patch_bin, r_bin, xi_bin, eta_bin)).transpose()

            # Assert that bins are sensible
            assert(not np.any(xi_bin >= grid_matrix.shape[2]))
            assert(not np.any(eta_bin >= grid_matrix.shape[3]))

        elif coordinate_system == grid.CoordinateSystem.cartesian:
            
            # Create grid
            grid_matrix = grid.create_cartesian_grid(max_radius=max_radius, n_features=n_features, bins_per_angstrom=bins_per_angstrom)

            # Bin each dimension of the cartesian coordinates
            indices = grid.discretize_into_cartesian_bins(xyz, max_radius, grid_matrix.shape)

            r = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)

        # Create an index array to keep track of entries within range
        if include_center:
            selector = np.where(r < max_radius)[0]
        else:
            # ALSO exclude features from residue itself 
            selector = np.where(np.logical_and(r < max_radius, res_indices[:,0] != residue_index))[0]

        # Apply selector on indices array
        indices = indices[selector]

        indices_rows = [tuple(row) for row in indices]
        duplicates = {}
        for index,row in enumerate(indices_rows):
            if indices_rows.count(row) > 1:
                index_matches = [index for index,value in enumerate(indices_rows) if value == row]
                index_matches.sort()
                if index_matches[0] not in duplicates:
                    duplicates[index_matches[0]] = index_matches
        if len(duplicates) > 0:
            print "WARNING: multiple atoms in same grid bin: (%s)" % pdb_id
            for duplicate_indices in duplicates.values():
                for i in duplicate_indices:
                    print "\t", features[selector][i]
                for i_index in range(len(duplicate_indices)):
                    coord1 = features[selector][duplicate_indices[i_index]][['x','y','z']].view(np.float32)
                    for j_index in range(i_index+1,len(duplicate_indices)):
                        coord2 = features[selector][duplicate_indices[j_index]][['x','y','z']].view(np.float32)
                        print '\t\tdistance(%s,%s) = %s' % (coord1, coord2,np.linalg.norm(coord2 - coord1))
                print
                
                
        # duplicate_indices = [index for index,row in enumerate(indices_rows) if indices_rows.count(row) > 1]
        # if len(duplicate_indices) > 0:
        #     print "WARNING: multiple atoms in same grid bin: "
        #     for i in duplicate_indices:
        #         print "\t", features[selector][i]
        #     for i_index in range(len(duplicate_indices)):
        #         coord1 = features[selector][duplicate_indices[i_index]][['x','y','z']].view(np.float32)
        #         for j_index in range(i_index+1,len(duplicate_indices)):
        #             coord2 = features[selector][duplicate_indices[j_index]][['x','y','z']].view(np.float32)
        #             print '\t\tdistance(%s,%s) = %s' % (coord1, coord2,np.linalg.norm(coord2 - coord1))
            # raise ValueError("multiple atoms in same grid bin")

        #print "Overlapping indicies:", np.sum(np.sum(np.diff(np.sort(indices, axis=0), axis=0), axis=1) == 0)
        
        # Append indices and selector for current residues to list
        indices_list.append(indices)
        selector_list.append(selector)


    # Data is stored most efficiently when encoded as Numpy arrays. Rather than storing the data
    # as a list, we therefore create a numpy array where the number of columns is set to match
    # that of the residue with most neighbors
    
    max_selector = max([len(selector) for selector in selector_list if selector is not None])
    selector_array = np.full((len(selector_list), max_selector), -1, dtype=np.int32)
    for i, selector in enumerate(selector_list):
        if selector is not None:
            selector_array[i,:len(selector)] = selector.astype(np.int32)
    # selector_array = np.zeros((len(selector_list), selector_list[0].shape[0]), dtype=np.bool_)
    # for i, selector in enumerate(selector_list):
    #     selector_array[i] = selector

    max_length = max([len(indices) for indices in indices_list if indices is not None])
    indices_list_shape_last_dim = None
    for indices in indices_list:
        if indices is not None:
            indices_list_shape_last_dim = indices.shape[1]
            break
    indices_array = np.full((len(indices_list), max_length, indices_list_shape_last_dim), -1, dtype=np.int16)
    for i, indices in enumerate(indices_list):
        if indices is not None:
            indices_array[i,:len(indices)] = indices.astype(np.int16)

    # # Save to disk
    # h5f = h5py.File(os.path.join(output_dir, "%s_residue_features.hp5"%pdb_id), 'w')
    # # h5f.create_dataset('features', data=feature_array, compression="gzip", compression_opts=9)
    # h5f.create_dataset('indices_array', data=indices_array, compression="gzip")
    # h5f.create_dataset('selector_array', data=selector_array, compression="gzip")
    # h5f.create_dataset('max_radius', data=max_radius)
    # h5f.create_dataset('n_features', data=n_features)
    # h5f.create_dataset('bins_per_angstrom', data=bins_per_angstrom)
    # h5f.close()

    # Save using numpy binary format
    np.savez_compressed(os.path.join(output_dir, "%s_residue_features"%pdb_id), indices=indices_array, selector=selector_array)
    

def extract_atomistic_features(pdb_filename, max_radius, n_features, bins_per_angstrom, add_seq_distance_feature, output_dir, coordinate_system, z_direction, include_center, dssp_executable):
    
    print pdb_filename
    # Read masses and charge information
    # try:
        # pdb_id, features, masses_array, charges_array, ss = extract_mass_charge(pdb_filename, dssp_executable)
    pdb_id, features, masses_array, charges_array, aa_one_hot, ss_one_hot, residue_index_array, chain_boundary_indices, chain_ids = extract_mass_charge(pdb_filename, dssp_executable)
    # except Exception as e:
    #     print "Error during parsing. Skipping. %s" % e
    #     return

    # Save protein level features
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez_compressed(os.path.join(output_dir, "%s_protein_features"%pdb_id),
                        masses=masses_array,
                        charges=charges_array,
                        residue_index=residue_index_array,
                        residue_features=["masses", "charges", 'residue_index'] if add_seq_distance_feature else ["masses", "charges"],
                        chain_boundary_indices = chain_boundary_indices,
                        chain_ids = chain_ids,
                        ss_one_hot = ss_one_hot,
                        aa_one_hot = aa_one_hot,
                        coordinate_system = np.array(coordinate_system.value, dtype=np.int32),
                        z_direction = np.array(z_direction.value, dtype=np.int32),
                        max_radius = np.array(max_radius, dtype=np.float32), # angstrom
                        n_features = np.array(n_features, dtype=np.int32),
                        bins_per_angstrom = np.array(bins_per_angstrom, dtype=np.float32),
                        n_residues=np.array(len(np.unique(features[['res_index']].view(int))), dtype=np.int32))

    # Embed in a grid
    embed_in_grid(features, pdb_id, output_dir,
                  max_radius=max_radius,
                  n_features=n_features,
                  bins_per_angstrom=bins_per_angstrom,
                  coordinate_system = coordinate_system,
                  z_direction = z_direction,
                  include_center = include_center)


def fetch_and_extract(line, max_radius, n_features, bins_per_angstrom, add_seq_distance_feature, reduce_executable, output_dir, pdb_output_dir, coordinate_system, z_direction, include_center, dssp_executable):

    import urllib
    url_base = "http://www.rcsb.org/pdb/download/downloadFile.do?fileFormat=pdb&compression=NO&structureId="
    
    entry = line.split()[0]
    if len(entry) != 5:
        print "skipping: %s" % (entry)
        return

    pdb_id = entry[:4]
    chain_id = entry[4]
    url = url_base+pdb_id

    try:
        structure = parse_pdb(urllib.urlopen(url), pdb_id, chain_id, reduce_executable)
        print "%s OK" % (entry)
        io = Bio.PDB.PDBIO()
        io.set_structure(structure)
        pdb_filename = os.path.join(pdb_output_dir, entry+'.pdb')
        io.save(pdb_filename)        

        extract_atomistic_features(pdb_filename, max_radius, n_features, bins_per_angstrom, add_seq_distance_feature, output_dir, coordinate_system, z_direction, include_center, dssp_executable)
    except Exception as e:
        print "%s Failed: %s" % (entry, e)
        pass

    
if __name__ == '__main__':
    import joblib

    import argparse
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help="sub-command help", dest="mode")
    parser_fecth = subparsers.add_parser("fetch-and-extract", help="fetch pdbs and extract features")
    parser_fecth.add_argument("input_filename", type=str)
    parser_fecth.add_argument("pdb_output_dir", type=str)
    parser_fecth.add_argument("output_dir", type=str)
    parser_fecth.add_argument("--dssp_executable", type=str)
    parser_fecth.add_argument("reduce_executable", type=str)

    parser_extract = subparsers.add_parser("extract", help="extract features")
    parser_extract.add_argument("pdb_input_dir", type=str)
    parser_extract.add_argument("output_dir", type=str)
    parser_extract.add_argument("--dssp_executable", type=str)

    parser.add_argument("--coordinate-system", choices=[e.name for e in grid.CoordinateSystem], default=grid.CoordinateSystem.spherical.name,
                        help="Which coordinate system to use (default: %(default)s)")
    parser.add_argument("--z-direction", choices=[e.name for e in grid.ZDirection], default=grid.ZDirection.outward.name,
                        help="Which direction to choose for z-axis (default: %(default)s)")
    parser.add_argument("--max-radius", metavar="VAL", type=int, default=12,
                        help="Maximal radius in angstrom (default: %(default)s)")
    parser.add_argument("--bins-per-angstrom", metavar="VAL", type=float, default=2,
                        help="Bins per Angstrom (default: %(default)s)")
    parser.add_argument("--n-proc", metavar="VAL", type=int, default=1,
                        help="Number of processes (default: %(default)s)")
    parser.add_argument("--include-center", action='store_true',
                        help="Include the center AA  (default: %(default)s)")
    parser.add_argument("--add-seq-distance-feature", action='store_true',
                        help="Add the sequence distance as a feature  (default: %(default)s)")

    args = parser.parse_args()

    print "# Arguments"
    for key, value in sorted(vars(args).iteritems()):
        print key, "=", value

    n_features = 2
    if args.add_seq_distance_feature:
        n_features = 3
    print "n_features: ", n_features
    
    coordinate_system = grid.CoordinateSystem[args.coordinate_system]
    z_direction = grid.ZDirection[args.z_direction]

    if args.mode == "fetch-and-extract":
        pdb_ids = [line.strip() for line in open(args.input_filename).readlines()]

        joblib.Parallel(n_jobs=args.n_proc, batch_size=1)(joblib.delayed(fetch_and_extract)(pdb_id, args.max_radius, n_features, args.bins_per_angstrom, args.add_seq_distance_feature, args.reduce_executable, args.output_dir, args.pdb_output_dir, coordinate_system = coordinate_system, z_direction = z_direction, include_center = args-include_center, dssp_executable = args.dssp_executable) for pdb_id in pdb_ids)
        
    elif args.mode == "extract":
        pdb_filenames = glob.glob(os.path.join(args.pdb_input_dir, "*.pdb"))
        joblib.Parallel(n_jobs=args.n_proc, batch_size=1)(joblib.delayed(extract_atomistic_features)(pdb_filename, args.max_radius, n_features, args.bins_per_angstrom, args.add_seq_distance_feature, args.output_dir, coordinate_system = coordinate_system, z_direction = z_direction, include_center = args.include_center, dssp_executable = args.dssp_executable) for pdb_filename in pdb_filenames)
    else:
        raise argparse.ArgumentTypeError("Unknown mode")


