"""Collect recent journal articles for task-specific manuscript positioning."""
from pathlib import Path
import concurrent.futures
import json
import re
import requests
from bs4 import BeautifulSoup

ROOT=Path('outputs/paper_ps/analysis/journal_recent_20261022')
JOURNALS={'GJI':'0956-540X','Computers_and_Geosciences':'0098-3004',
          'SRL':'0895-0695','Earth_and_Space_Science':'2333-5084'}
QUERIES=('seismic phase picking deep learning','earthquake arrival time frequency transformer')


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    jobs=[(name,issn,q) for name,issn in JOURNALS.items() for q in QUERIES]
    def search(job):
        name,issn,query=job
        try:
            r=requests.get(f'https://api.crossref.org/journals/{issn}/works',params={
                'query':query,'filter':'from-pub-date:2024-01-01,until-pub-date:2026-10-22,type:journal-article',
                'rows':12},timeout=30,headers={'User-Agent':'SeismicManuscriptResearch/1.0'})
            if r.status_code!=200:return dict(journal=name,query=query,status=r.status_code,url=r.url,items=[])
            items=[]
            for j in r.json()['message']['items']:
                abstract=BeautifulSoup(j.get('abstract',''),'html.parser').get_text(' ',strip=True)
                items.append({**j,'abstract_text':abstract})
            return dict(journal=name,query=query,status=200,url=r.url,items=items)
        except Exception as e:return dict(journal=name,query=query,error=str(e),items=[])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(search,jobs))
    (ROOT/'crossref_searches.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    seen={}
    for search_result in results:
        for j in search_result['items']:
            title=' '.join(j.get('title',[]));abstract=j['abstract_text']
            if re.search(r'pick|arrival|phase|earthquake|seismic',title,re.I):
                seen[j['DOI']]=dict(journal=search_result['journal'],doi=j['DOI'],title=title,
                    date=j.get('published',{}).get('date-parts'),abstract=abstract,URL=j.get('URL'),
                    links=j.get('link'),authors=j.get('author'),volume=j.get('volume'),page=j.get('page'),
                    published_online=j.get('published-online'),published_print=j.get('published-print'))
    (ROOT/'candidate_articles.json').write_text(json.dumps(list(seen.values()),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([dict(journal=r['journal'],status=r.get('status'),count=len(r['items'])) for r in results]))
    for j in seen.values():
        print(j['journal'],j['date'],j['doi'],j['title'],'abstract_chars',len(j['abstract']))


if __name__=='__main__':run()
