import generic
import spacy_streamlit
import streamlit as st

def show_layout(type='page',data=None,layout=[.2,.9]):
    cols = st.columns(layout)
    returns = []

    if type == 'page':
        data = ['Prev Text','Next Text']

    for col_idx, col in enumerate(cols):
        with col:
            # if type == 'page':
            #     returns.append(st.button(data[col_idx],key=data[col_idx].lower().replace(' ','_')))
            if data:
                returns.append(st.button(data[col_idx],key=data[col_idx].lower().replace(' ','_')))                

    return returns


def display_sidebar():
    with st.sidebar:
        # # Data
        # generic.read_data()
        texts = st.session_state.data['doc'][st.session_state.page]

        if len(texts) > 0:
            # groups = list(texts.spans.values())
            st.subheader('Coref groups')
            if st.session_state.data['coref']:
                corefs = st.session_state.data['coref']
                # groups = list(map(lambda idx:st.multiselect(label=f'Group #{idx+1}',options=keywords[idx],key=f'multiselect_{idx}',default=keywords[idx]),range(len(keywords))))
                # corefs = list(map(lambda idx:st.multiselect(label=f'Group #{idx+1}',options=map(lambda x:f'{x}: {str(corefs[idx][x])}',range(len(corefs[idx]))),key=f'multiselect_{idx}',default=map(lambda x:f'{x}: {str(corefs[idx][x])}',range(len(corefs[idx])))),range(len(corefs))))
                corefs = [st.multiselect(label=f"Coref #{idx}",options=map(lambda x:f'{x}: {str(items[x])}',range(len(items))),key=f'multiselect_{idx}',default=map(lambda x:f'{x}: {str(items[x])}',range(len(items)))) for idx,items in corefs.items()]
                
            # return groups

        # return None


def display_spacy(doc,labels):
    colors = ['#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']
    symbol_cnt = len(list(filter(lambda label:label.lower().find('group')==-1,labels)))
    colors_dict = {label:colors[0] if label.lower().find('group')!=-1 else colors[label_idx-symbol_cnt] for label_idx, label in enumerate(labels)}

    spacy_streamlit.visualize_ner(doc=doc,labels=labels,colors=colors_dict,show_table=False,title='',manual=True)


def process_iterator(iter_obj,page_num,groups):
    text_idx, line = generic.check_iterator(iter_obj,page_num)

    if len(line) > 0 and groups:
        st.markdown(f'Current Page: `{page_num+1}` of `{len(iter_obj)}`')
        generic.update_session(session_key='page',value=page_num)
        sel_text, symbol_dict = generic.extract_text(line,groups)
        doc, labels = generic.process_displayc(sel_text,groups,symbol_dict)
        sel_text['symbol'] = [symbol for symbol in labels if symbol.lower().find('group')==-1]
        sel_text['name'] = st.session_state.data['quotes'].loc[st.session_state.data['quotes']['symbol'].isin(sel_text['symbol']),'name'].tolist()
        # show_table(sel_text)
        display_spacy(doc,labels)

        return True

    return False


def display_texts(pages,groups=None,page_num=0):
    prev_page, next_page, page_num = generic.process_btn(pages,page_num)
    iter_obj = st.session_state.data['doc']

    texts = st.session_state.data['doc'][st.session_state.page]    
    if len(texts) > 0:
        groups = list(texts.spans.values())

    update_status = process_iterator(iter_obj,page_num,groups)

    return prev_page, next_page, update_status