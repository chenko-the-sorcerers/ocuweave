"""Preservation and transitive-group guards use synthetic identities only."""
import copy
import unittest
from ocuweave.data.splitting.development import regroup


def source():
    return {'patient_identity_verified':False,'records':[
        {'sample_id':str(i),'split':s,'split_group':str(i),'pixel_sha256':str(i),'similarity_group':str(i),
         'patient_id':None,'label':'healthy'} for i,s in enumerate(('train','validation','validation','validation','test'))]}


def candidate(a,b,sa='train',sb='validation'):
    return {'a':str(a),'b':str(b),'a_split':sa,'b_split':sb}


class DevelopmentFreezeTests(unittest.TestCase):
    def test_transitive_groups_keep_all_records_and_test(self):
        original=source();before=copy.deepcopy(original)
        result,decision=regroup(original,{'pairs':[candidate(0,1),candidate(1,2,'validation')],'truncated':False})
        self.assertEqual(original,before)
        self.assertEqual(result['records'][-1],original['records'][-1])
        self.assertEqual([r['split'] for r in result['records']],['train','train','train','validation','test'])
        self.assertEqual(len({r['split_group'] for r in result['records'][:3]}),1)
        self.assertTrue(all(r['patient_id'] is None for r in result['records']))
        self.assertEqual(decision['quality_exclusions'],0)
        self.assertEqual(len(result['records']),len(original['records']))

    def test_fail_closed_for_test_candidate_and_stale_audit(self):
        for pairs in ([candidate(0,4,'train','test')],[candidate(0,1,'validation')], [candidate(0,99)]):
            with self.assertRaises(ValueError):regroup(source(),{'pairs':pairs,'truncated':False})
        with self.assertRaises(ValueError):regroup(source(),{'pairs':[],'truncated':True})

    def test_existing_group_bridge_is_preserved(self):
        original=source();original['records'][1]['similarity_group']=original['records'][2]['similarity_group']='family'
        result,_=regroup(original,{'pairs':[candidate(0,1)],'truncated':False})
        self.assertEqual([r['split'] for r in result['records'][:3]],['train']*3)
        original['records'][4]['similarity_group']='family'
        with self.assertRaises(ValueError):regroup(original,{'pairs':[candidate(0,1)],'truncated':False})
