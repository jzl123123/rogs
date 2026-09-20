import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from rogs.hugsim.config import load_profile, training_config
from rogs.hugsim.data import camera_geometry, convert_labels, pose_matrix, project_ground, prepare, validate_source
from verify_hugsim import verify, definitions


class HugsimTests(unittest.TestCase):
    def test_bev_world_axes_and_pixel_budget(self):
        from rogs.hugsim.export import bev_geometry
        points=np.array([[10,20,1],[12,24,2]],dtype=float)
        pose,k,w,h,meta=bev_geometry(points,.5,margin=0)
        self.assertEqual((w,h),(4,8))
        world=np.array([10.25,23.75,1,1])
        cam=np.linalg.inv(pose)@world
        np.testing.assert_allclose(k[:2,:2]@cam[:2],[.5,.5])
        self.assertAlmostEqual(meta['camera_z']-cam[2],world[2])
        with self.assertRaisesRegex(ValueError,'pixels'):
            bev_geometry(points,.01,max_pixels=100)

    def test_upstream_algorithms_preserved(self):
        self.assertEqual(verify(),11)
        self.assertNotEqual(definitions('def f(x): return x * 2'),definitions('def f(x): return x * 3'))

    def test_training_defaults_and_override_isolation(self):
        p=load_profile(ROOT/'configs/hugsim/shared_nusc.yaml')
        cfg=training_config(p,'/data/shared','/runs/test')
        self.assertEqual(cfg['train']['iterations'],30000)
        self.assertEqual(cfg['ground'],{'n_sample':10,'grid_len':0.2,'min':-2,'range':4})
        self.assertEqual(cfg['opt']['lambda_dssim'],0.2)
        before=copy.deepcopy(p)
        cfg['ground']['range']=999
        self.assertEqual(p,before)
        p['overrides']={'train':{'iterations':10}}
        with self.assertRaisesRegex(ValueError,'checkpoints'):training_config(p,'a','b')

    def test_projection_survives_resize_crop_and_coordinate_change(self):
        k=np.array([[100.,0,80],[0,100,45],[0,0,1]])
        c2w=pose_matrix([np.cos(.25),0,0,np.sin(.25)],[4,5,2])
        inv=np.linalg.inv(pose_matrix([1,0,0,0],[10,20,30]))
        new_k,new_pose=camera_geometry(k,c2w,inv,(160,90),(80,45),22)
        p_camera=np.array([.4,.2,3.,1.])
        world=c2w@p_camera
        transformed=inv@world
        p_new=np.linalg.inv(new_pose)@transformed
        pixel=k@p_camera[:3];pixel=pixel[:2]/pixel[2]
        actual=new_k@p_new[:3];actual=actual[:2]/actual[2]
        np.testing.assert_allclose(actual,pixel*.5-[0,22],atol=1e-10)
        np.testing.assert_array_equal(k,[[100,0,80],[0,100,45],[0,0,1]])

    def test_mapillary_masks_preserve_rogs_ground_and_back_camera_cut(self):
        label=np.full((100,100),13,dtype=np.uint8)
        label[30:40]=15
        label[50:60]=29
        out=convert_labels(label,'CAM_BACK','rogs',50)
        self.assertTrue(np.all(out[:10]==0))  # RoGS includes terrain
        self.assertTrue(np.all(out[33:]==19))
        self.assertTrue(np.all(convert_labels(label,'CAM_FRONT','rogs',0)[30:40]==1))
        self.assertTrue(np.all(convert_labels(label,'CAM_FRONT','road_sidewalk',0)[50:60]==19))
        self.assertTrue(np.all(label[30:40]==15))
        with self.assertRaises(ValueError):convert_labels(np.array([[255]],dtype=np.uint8),'CAM_FRONT','rogs',0)

    def test_nearest_camera_ground_projection(self):
        poses=np.stack([np.eye(4)]*3)
        poses[1,0,3]=10;poses[2,0,3]=100
        points=np.array([[.1,4,3],[9.9,6,4],[100,3,2]])
        result=project_ground(points,poses,1.6)
        np.testing.assert_allclose(result[:,1],1.6)
        np.testing.assert_allclose(result[:,[0,2]],points[:,[0,2]])

    def test_nested_sparse_archive_preserves_selected_sources(self):
        from fetch_thirdparty import extract
        with tempfile.TemporaryDirectory() as d:
            archive=Path(d)/'a.zip';out=Path(d)/'out'
            with zipfile.ZipFile(archive,'w') as z:
                for name in ['root/LICENSE','root/data/nusc/a.py','root/data/other/b.py','root/scene/c.py']:
                    z.writestr(name,'test')
            extract(archive,out,['data/nusc','scene'])
            self.assertTrue((out/'LICENSE').is_file())
            self.assertTrue((out/'data/nusc/a.py').is_file())
            self.assertFalse((out/'data/other').exists())

    def test_real_image_preparation_cache_and_stale_input_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);meta=root/'nusc/v1.0-trainval';meta.mkdir(parents=True)
            ds={'dataset':'NuscDataset','base_dir':str(root/'nusc'),'image_dir':str(root/'seg'),
                'road_gt_dir':str(root/'ground'),'version':'trainval','clip_list':['scene-test'],
                'camera_names':['CAM_FRONT'],'image_width':8,'image_height':8}
            samples=[];sd=[]
            # Five keyframes permit a one-frame holdout while preserving exact RoGS indexing.
            for i in range(5):
                samples.append({'token':f's{i}','scene_token':'scene','timestamp':i*100})
                sd.append({'token':f'c{i}','sample_token':f's{i}','is_key_frame':True,
                           'next':f'c{i+1}' if i<4 else '', 'timestamp':i*100,
                           'filename':f'samples/CAM_FRONT/{i}.jpg','ego_pose_token':f'e{i}',
                           'calibrated_sensor_token':'cal'})
                image=root/'nusc'/sd[-1]['filename'];image.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.full((16,16,3),100+i,dtype=np.uint8)).save(image)
                label=root/f'seg/samples/seg_CAM_FRONT/{i}.png';label.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.full((16,16),13,dtype=np.uint8)).save(label)
            tables={'scene':[{'token':'scene','name':'scene-test'}],'sample':samples,'sample_data':sd,
                    'sensor':[{'token':'sensor','channel':'CAM_FRONT'}],
                    'calibrated_sensor':[{'token':'cal','sensor_token':'sensor','rotation':[1,0,0,0],
                                         'translation':[0,0,0],'camera_intrinsic':[[10,0,8],[0,10,8],[0,0,1]]}],
                    'ego_pose':[{'token':f'e{i}','rotation':[1,0,0,0],'translation':[i,0,0]} for i in range(5)]}
            for name,table in tables.items():(meta/(name+'.json')).write_text(json.dumps(table))
            (root/'ground').mkdir()
            vertices=np.zeros(4,dtype=[(k,'f4') for k in ('x','y','z','r','g','b','label')])
            vertices['x']=[0,1,2,3];vertices['z']=3
            for k in ('r','g','b'):vertices[k]=.5
            PlyData([PlyElement.describe(vertices,'vertex')]).write(root/'ground/scene-test.ply')
            profile=load_profile(ROOT/'configs/hugsim/shared_nusc.yaml')
            profile['prepared_root']=str(root/'prepared');profile['initialization']['camera_height']=1.6
            cfg={'dataset':ds,'seed':17}
            with contextlib.redirect_stdout(io.StringIO()):
                source=prepare(cfg,profile)
                self.assertEqual(prepare(cfg,profile),source)
            self.assertEqual(validate_source(source,True),{'train':4,'test':1})
            frames=json.loads((source/'meta_data.json').read_text())['frames']
            self.assertEqual((frames[0]['width'],frames[0]['height']),(8,4))
            self.assertEqual(frames[0]['intrinsics'][1][2],0)
            self.assertEqual([f['source']['token'] for f in frames],[f'c{i}' for i in range(5)])
            self.assertEqual(frames[-1]['split'],'test')
            cloud=PlyData.read(source/'ground_points3d.ply')['vertex']
            np.testing.assert_allclose(cloud['y'],1.6)
            Image.fromarray(np.full((16,16),15,dtype=np.uint8)).save(root/'seg/samples/seg_CAM_FRONT/0.png')
            with self.assertRaisesRegex(ValueError,'differs'):prepare(cfg,profile)

if __name__=='__main__':unittest.main()
