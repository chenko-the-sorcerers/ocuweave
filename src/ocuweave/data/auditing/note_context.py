"""Conservative clause-local context flags. Not a validated assertion resolver."""
from collections import Counter
import re
from ocuweave.data.auditing.clinical_notes import PATTERNS

DIAGNOSIS=re.compile(PATTERNS['diagnosis_term'],re.I)
CONTEXT={
 'negation_near_mention':r'\b(?:no|not|without|denies|negative\s+for|unlikely)\b',
 'family_near_mention':r'\b(?:family|fhx|f/h|mother|father|sister|brother)\b',
 'suspected_near_mention':r'\b(?:suspect\w*|possible|probable|risk|rule\s+out|r/o)\b',
 'historical_near_mention':r'\b(?:history|hx|previous\w*|prior|past|known)\b',
 'laterality_near_mention':r'\b(?:od|os|ou|right\s+eye|left\s+eye|both\s+eyes)\b'}


def context_flags(note):
    mentions=[]
    # Split at sentence/newline/semicolon or contrast cues. Periods in abbreviations
    # and shorthand can still cause errors; we never call a clause 'cleared'.
    for clause in re.finditer(r'[^.;\n]+',note):
        for segment in re.finditer(r'.+?(?=\b(?:but|however|although)\b|$)',clause.group(),re.I):
            text=segment.group();offset=clause.start()+segment.start()
            for match in DIAGNOSIS.finditer(text):
                left=list(re.finditer(r'\S+',text[:match.start()]))[-6:]
                start=left[0].start() if left else match.start()
                right=list(re.finditer(r'\S+',text[match.end():]))[:3]
                end=match.end()+(right[-1].end() if right else 0)
                window=text[start:end]
                flags={k:bool(re.search(pattern,window,re.I)) for k,pattern in CONTEXT.items()}
                mentions.append({'start':offset+match.start(),'end':offset+match.end(),**flags,
                    'unqualified_mention_proxy':not any(flags[k] for k in ('negation_near_mention','family_near_mention','suspected_near_mention','historical_near_mention'))})
    return {'mentions':mentions,
        'treatment_proxy':bool(re.search(PATTERNS['treatment_proxy'],note,re.I)),
        'endpoint_proxy':bool(re.search(PATTERNS['endpoint_proxy'],note,re.I)),
        'date_token_present':bool(re.search(r'\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|20\d{2})\b',note)),
        'prediction_time_verified':False,'clinical_assertion_review_complete':False}


def review_context(rows):
    counts={};queue=[]
    for row in rows:
        if row['use'] not in {'training','validation'}:continue
        flags=context_flags(row['note'])
        record={'filename':row['filename'],'split':row['use'],**flags,'review_status':'unreviewed'}
        record['requires_review']=bool(flags['mentions'] or flags['treatment_proxy'] or flags['endpoint_proxy'])
        queue.append(record)
        count=counts.setdefault(row['use'],Counter());count['rows']+=1
        count['notes_with_diagnosis_mention']+=bool(flags['mentions'])
        for key in (*CONTEXT,'unqualified_mention_proxy'):
            count[key]+=any(m[key] for m in flags['mentions'])
        for key in ('treatment_proxy','endpoint_proxy','date_token_present'):count[key]+=flags[key]
        count['review_priority_flagged']+=record['requires_review']
    return {'protocol':'fairvlmed-context-review-v1','counts':counts,'review_queue':queue,
        'test_note_content_analyzed':False,'raw_text_exported':False,'training_cleared':False,
        'temporal_order_verified':False,'limitations':[
            'Clause-local proximity flags can misread negation, abbreviations, experiencer and temporality.',
            'A date token is not a verified encounter or prediction time.',
            'Unflagged notes are not declared safe. No note is redacted, deleted or used for training.']}
