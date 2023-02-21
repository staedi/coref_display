import pandas as pd
import numpy as np
import pickle
import streamlit as st
import re
import ast

def init_session(session_key=None):
    if not session_key:
        if 'page' not in st.session_state:
            st.session_state['page'] = 0
        if 'region' not in st.session_state:
            st.session_state['region'] = {'US':None,'KR':None}
        if 'file_path' not in st.session_state:
            st.session_state['file_path'] = {'quotes':{'dir':None,'file':None},'doc':{'dir':None,'file':None}}
        # if 'git_path' not in st.session_state:
        #     st.session_state['git_path'] = {'api':None,'owner':None,'repo':None}
        # if 'git_header' not in st.session_state:
        #     st.session_state['git_header'] = {'accept':None,'authorization':None}
        # # if 'nlp' not in st.session_state:
        # #     st.session_state['nlp'] = {'US':{'core':None,'coref':None},'KR':{'core':None,'coref':None}}
        if 'data' not in st.session_state:
            st.session_state['data'] = {'doc':None,'quotes':pd.DataFrame(),'coref':None}

    else:
        if session_key == 'page':
            st.session_state['page'] = 0
        if session_key == 'region':
            st.session_state['region'] = {'US':'','KR':''}
        elif st.session_state == 'file_path':
            st.session_state['file_path'] = {'quotes':{'dir':None,'file':None},'doc':{'dir':None,'file':None}}
        # elif st.session_state == 'git_path':
        #     st.session_state['git_path'] = {'api':None,'owner':None,'repo':None}
        # elif st.session_state == 'git_header':
        #     st.session_state['git_header'] = {'accept':None,'authorization':None}
        # # elif st.session_state == 'nlp':
        # #     st.session_state['nlp'] = {'US':{'core':None,'coref':None},'KR':{'core':None,'coref':None}}
        elif st.session_state == 'data':
            st.session_state['data'] = {'doc':None,'quotes':pd.DataFrame(),'coref':None}


def update_session(session_key,value,key=None):
    if session_key in st.session_state:
        # selectbox for Title
        if session_key == 'select_title':
            st.session_state.select_title = None

        # File name for region 
        elif session_key == 'region':
            if key != None:
                st.session_state.region[key] = value
            else:
                st.session_state.region = value
        # Directory and File paths
        elif session_key == 'file_path':
            if key != None:
                st.session_state.file_path[key] = value
            else:
                st.session_state.file_path = value
        # Git Crendentials and paths
        elif session_key == 'git_path':
            if key != None:
                st.session_state.git_path[key] = value
            else:
                st.session_state.git_path = value
        # Header for Git Request
        elif session_key == 'git_header':
            if key != None:
                st.session_state.git_header[key] = value
            else:
                st.session_state.git_header = value
        # # spaCy model
        # elif session_key == 'nlp':
        #     if key != None:
        #         st.session_state.nlp[key] = value
        #     else:
        #         st.session_state.nlp = value
        # Page
        elif session_key == 'page':
            st.session_state.page = value
        # Data
        elif session_key == 'data':
            if key != None:
                st.session_state.data[key] = value
            else:
                st.session_state.data = value


def init_params():
    update_session(session_key='region',value={'US':'US_topics'})
    update_session(session_key='file_path',value={'quotes':{'dir':st.secrets.data_dir.quotes,'file':'quotes.csv'},'doc':{'dir':st.secrets.data_dir.doc,'file':'coref_pickle.pkl'}})
    # update_session(session_key='git_path',value={'api':st.secrets.git.api,'owner':st.secrets.git.owner,'repo':st.secrets.git.repo})
    # update_session(session_key='git_header',value={'accept':st.secrets.header.accept,'authorization':st.secrets.header.authorization})


def read_data():
    if not st.session_state.data['doc']:
        regions, file_paths = st.session_state.region, st.session_state.file_path
        data = dict.fromkeys(file_paths.keys())

        for keys, paths in file_paths.items():
            for region_atr in regions.values():
                if keys == 'quotes':
                    path = f"{paths['dir']}{region_atr}_{paths['file']}"
                    df = pd.read_csv(path,sep=",",encoding='utf-8-sig',dtype={'article_id':object,'symbol':object,'summary':object})
                    data[keys] = pd.concat([data[keys],df])
                    data[keys][['symbol','name']] = data[keys][['symbol','name']].applymap(lambda x:str(x))
                    cols = ['region','symbol','name','summary']
                    data[keys] = data[keys][cols]
                    data[keys]['symbol'] = data[keys]['symbol'].apply(lambda x:x[1:] if re.search(r'\b[A-Z]\d+\b',x) else x)
                    data[keys]['summary'] = data[keys]['summary'].apply(lambda x:ast.literal_eval(x))
                    data[keys]['summary_compact'] = data[keys]['summary'].apply(lambda x:x['company']+x['details'])

                elif keys == 'doc':
                    path = f"{paths['dir']}{paths['file']}"
                    with open(path,'rb') as path:
                        df = pickle.load(path)
                    data[keys] = pickle.loads(df)

            # Update to session data
            update_session(session_key='data',key=keys,value=data[keys])


def extract_text(sel_article,keywords):
    sel_df = sel_article #st.session_state.data['doc'][st.session_state.page]
    group_cols = ['region','date','provider','link','headline','text']

    if re.search(r'^[ㄱ-ㅎ가-힣]',str(sel_df)):
        region = 'KR'
    else:
        region = 'US'

    sel_text = {'region':region,'text':sel_df}
    symbol_df = st.session_state.data['quotes'].loc[(st.session_state.data['quotes']['region']==sel_text['region']),['symbol','summary_compact']]
    symbol_dict = {symbol:summary for symbol,summary in zip(symbol_df['symbol'],symbol_df['summary_compact'])}

    # # sel_text = {type:sel_df[type][0] for type in group_cols}
    # # sel_text['symbol'] = sel_df['symbol'].tolist()
    # # sel_text['name'] = sel_df['name'].tolist()

    keywords = [keyword for group in keywords for keyword in group]
    symbol_dup_dict = {symbol:filter(lambda x:list(filter(lambda keyword:str(keyword).find(x)!=-1,keywords)),summary) for symbol,summary in symbol_dict.items()}
    symbol_dict = {symbol:summary for symbol,summary in symbol_dict.items() if summary not in symbol_dup_dict[symbol]}

    # # if re.search(r'^[ㄱ-ㅎ가-힣]',sel_article['headline']):
    # #     nlp = st.session_state.nlp['KR']['core']
    # # else:
    # #     nlp = st.session_state.nlp['KR']['core']
    # # return text, nlp
    return sel_text, symbol_dict


def get_term_idx(keywords,rep_groups):
    keywords = [list(map(lambda x:x.text.lower(),keywords[group])) for group in range(len(keywords))] #if group not in rep_groups.keys()]
    terms_idx = {keyword.lower():None for group in keywords for keyword in group}
    for term in terms_idx:
        terms_idx[term] = [len(np.where(np.array(keyword)==term.lower())[0])>0 for idx,keyword in enumerate(keywords)].index(1)

    return terms_idx


def process_coref(text_dict,keywords,symbol_dict):
    region, text = text_dict['region'], text_dict['text']
    text_copy = text.text

    # Symbols ({symbol:spans})
    symbol_pre_spans = {symbol:set(filter(lambda x:len(list(re.finditer(rf"\b{x.lower()}(\W?)\b",text.text.lower()) if region != 'KR' else re.finditer(rf"\b{x.lower()}(\w?)\b",text.text.lower())))>0,summary)) for symbol,summary in symbol_dict.items()}
    symbol_re_spans = {symbol:set(map(lambda x:re.finditer(rf"\b{x.lower()}(\W?)\b",text.text.lower()) if region != 'KR' else re.finditer(rf"\b{x.lower()}(\w?)\b",text.text.lower()),summary)) for symbol,summary in symbol_pre_spans.items()}
    symbol_re_spans = {symbol:spans for symbol,spans in symbol_re_spans.items() if len(spans)>0}        
    symbol_spans = {symbol:sorted(set([span.span() for spans in summary for span in spans])) for symbol,summary in symbol_re_spans.items()}

    # Coref Groups ({group:symbol[name]})
    # symbol_key_pre_groups = {symbol:list(map(lambda summary:list(filter(lambda group:list(filter(lambda keyword:text[keyword.start:keyword.start+1].text==summary or text[keyword.start:keyword.start+1].text==summary.split()[0] or text[keyword.end-1:keyword.end].text==summary.split()[-1],keywords[group])),range(len(keywords)))), set(map(lambda span:text.text[span[0]:span[-1]].strip(),spans)))) for symbol,spans in symbol_spans.items()}
    # symbol_key_pre_groups = {symbol:list(map(lambda summary:list(filter(lambda group:text[keywords[group][0].start:keywords[group][0].start+1].text==summary or text[keywords[group][0].start:keywords[group][0].start+1].text==summary.split()[0] or text[keywords[group][0].end-1:keywords[group][0].end].text==summary.split()[-1], range(len(keywords)))), set(map(lambda span:text.text[span[0]:span[-1]].strip(),spans)))) for symbol,spans in symbol_spans.items()}
    symbol_key_pre_groups = {symbol:list(map(lambda summary:list(filter(lambda group:summary in keywords[group][0].text and len(keywords[group][0].text.strip().split())<5, range(len(keywords)))), set(map(lambda span:text.text[span[0]:span[-1]].strip(),spans)))) for symbol,spans in symbol_spans.items()}
    symbol_key_groups = {symbol:sorted(set([group for groups in idx for group in groups])) for symbol,idx in symbol_key_pre_groups.items()}
    # symbol_key_groups = {symbol:set(filter(lambda group:list(filter(lambda keyword:text[keyword.start:keyword.start+1].text==text.text[span[0]:span[-1]].strip().split()[0] or text[keyword.end-1:keyword.end].text==text.text[span[0]:span[-1]].strip().split()[-1],keywords[group])),range(len(keywords)))) for symbol,spans in symbol_spans.items() for span in spans}
    key_symbol_spans = {key:sorted(set(map(lambda spans:text.text[spans[0]:spans[-1]].strip(),symbol_spans[symbol]))) for symbol,group in symbol_key_groups.items() for key in group if group}

    rep_groups = key_symbol_spans

    ## Manipulate Coref Groups
    # Initialize
    offset = 0
    reindex = []

    # Check for spans needing replacement
    for chain, rep_targets in key_symbol_spans.items():
        for span in keywords[chain]:
            if span.text not in rep_targets:
                reindex.append([span.start_char,span.end_char,rep_targets[0]])

    re_index = []

    for idx, spans in enumerate(sorted(reindex, key=lambda x:x[0])):
        if idx == 0 or (spans[0]>sorted(reindex, key=lambda x:x[0])[idx-1][0] and spans[1]>sorted(reindex, key=lambda x:x[0])[idx-1][1]):
            re_index.append(spans)

    # Actual replacement happens here
    for span in sorted(re_index, key=lambda x:x[0]):
        text_copy = text_copy[0:span[0]+offset]+span[2]+text_copy[span[1]+offset:]
        offset += len(span[2]) - (span[1] - span[0])

    # Update to session data
    update_session(session_key='data',key='coref',value={group:keywords[group] for group in rep_groups})

    return text_copy, rep_groups


def match_pattern(text_dict,keywords,symbol_dict,rep_groups):
    region, text = text_dict['region'], text_dict['text']
    
    # Keywords
    terms = [keyword for group in range(len(keywords)) for keyword in keywords[group] if group not in rep_groups.keys()]
    # key_re_spans = set(map(lambda x:re.finditer(rf"\b{x.text.lower()}",text.lower()) if region != 'KR' else re.finditer(rf"{x.text.lower()}",text.lower()),terms))
    # key_spans = sorted(set([span.span() for spans in key_re_spans for span in spans]))
    key_spans = sorted(set([(keyword.start_char,keyword.end_char) for keyword in terms]))

    # Symbols
    symbol_pre_spans = {symbol:set(filter(lambda x:len(list(re.finditer(rf"\b{x.lower()}(\W?)\b",text.lower()) if region != 'KR' else re.finditer(rf"\b{x.lower()}(\w?)\b",text.lower())))>0,summary)) for symbol,summary in symbol_dict.items()}
    symbol_re_spans = {symbol:set(map(lambda x:re.finditer(rf"\b{x.lower()}(\W?)\b",text.lower()) if region != 'KR' else re.finditer(rf"\b{x.lower()}(\w?)\b",text.lower()),summary)) for symbol,summary in symbol_pre_spans.items()}
    symbol_re_spans = {symbol:spans for symbol,spans in symbol_re_spans.items() if len(spans)>0}
    # symbol_spans = {symbol:sorted(set([span.span() for spans in summary for span in spans if span.span() not in key_spans])) for symbol,summary in symbol_re_spans.items()}
    symbol_spans = {symbol:sorted(set([span.span() for spans in summary for span in spans if len(set(filter(lambda key_span:key_span[0]<=span.span()[0] and key_span[1]>=span.span()[1],key_spans)))==0  ])) for symbol,summary in symbol_re_spans.items()}

    # Filter duplicating symbols
    symbol_spans = {symbol:{span[0]:span[-1] for span in spans} for symbol,spans in symbol_spans.items()}
    symbol_spans = {symbol:{end:start for start,end in spans.items()} for symbol,spans in symbol_spans.items()}
    symbol_spans = [{symbol:[start,end]} for symbol,spans in symbol_spans.items() for end,start in spans.items()]

    return key_spans, symbol_spans


def get_obj_value(iter_obj,target_value,access='key'):
    if access == 'key':    # access by key
        # target_list = target_value.replace('(',' - ').replace(')',' - ').split(' - ')
        target_list = re.split(r'(\s\()(\d+)(\))',target_value)
        
        if len(target_list)==1:
            target_list.append(0)
        else:
            target_list = [target_list[0],target_list[2]]
            target_list[1] = int(target_list[1])-1

        return iter_obj.get(target_list[0])[target_list[1]]
        # return iter_obj.get(target_value)
        # return max([iter[1] for iter in iter_obj if iter[0]==target_value])
    else:   # access by value
        # return max([f'{value.index(target_value)+1}: {idx}' if len(value)>1 else idx for idx,value in iter_obj.items() if target_value in value])
        try:
            return max([idx for idx,value in iter_obj.items() if target_value in value])
            # return max([f'{idx} ({value.index(target_value)+1})' if len(value)>1 else idx for idx,value in iter_obj.items() if target_value in value])
        except:
            return None


def check_iterator(iter_obj,page_num):
    try:
        text_idx, line = page_num, iter_obj[page_num] #list(iter_obj)[page_num]
        return text_idx, line
    except:
        return None, ''


def process_displayc(text_dict,keywords,symbol_dict=None):
    # Manipulate Coref Groups
    text_dict['text'], rep_groups = process_coref(text_dict,keywords,symbol_dict)

    # Get outer index
    terms_idx = get_term_idx(keywords,rep_groups)
    # Find spans (Keywords, Symbol)
    key_spans, symbol_spans = match_pattern(text_dict,keywords,symbol_dict,rep_groups)

    region, text = text_dict['region'], text_dict['text']

    # Spans for manual Doc
    # Keywords
    spans_key = [{'start':span[0],'end':span[1],'label':f'Group #{terms_idx[text[span[0]:span[1]].lower()]+1}'} for span in key_spans if terms_idx.get(text[span[0]:span[1]].lower())]
    # Symbol
    spans_symbol = [{'start':span[0],'end':span[1],'label':symbol} if not get_obj_value(rep_groups,text[span[0]:span[1]].strip(),'value') else {'start':span[0],'end':span[1],'label':f"Coref #{get_obj_value(rep_groups,text[span[0]:span[1]].strip(),'value')}"} for spans in symbol_spans for symbol,span in spans.items()]

    spans = spans_key+spans_symbol

    ## Labeling
    # labels = sorted(set([span['label'] for span in spans]),key=lambda x:x.lower().find('group')!=-1 and x in map(lambda label:label['label'],spans_symbol))
    labels = sorted(set([span['label'] for span in spans]),key=lambda x:x.lower().find('group')!=-1)

    # Make output Doc
    doc = [{'text':text,'ents':spans}]

    return doc, labels


def process_btn(pages,page_num=0):
    if pages[0]:    # prev_page
        if st.session_state.page > 0:
            update_session(session_key='page',value=st.session_state.page-1)
            update_session(session_key='data',key='coref',value=None)

    if pages[1]:    # next_page
        if st.session_state.page < len(st.session_state.data['doc'])-1:
            update_session(session_key='page',value=st.session_state.page+1)
            update_session(session_key='data',key='coref',value=None)

    page_num = st.session_state.page
    prev_page, next_page = False, False

    return prev_page, next_page, page_num