"""공개 HTML 정리. 기존 수집기의 변환 규칙을 독립 모듈로 보존한다."""
import html
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup


def html_text(value):
    soup=BeautifulSoup(value,'html.parser')
    for tag in soup(['script','style']):
        tag.decompose()
    for table in soup.find_all('table'):
        rows=[[cell.get_text(' ',strip=True).replace('|','/') for cell in row.find_all(['th','td'])]
              for row in table.find_all('tr')]
        rows=[row for row in rows if row]
        if rows:
            lines=['| '+' | '.join(row)+' |' for row in rows]
            lines.insert(1,'| '+' | '.join(['---']*len(rows[0]))+' |')
            table.replace_with('\n'+'\n'.join(lines)+'\n')
    for tag in soup.find_all(['h1','h2','h3','h4']):
        tag.insert_before('\n'+'#'*int(tag.name[1])+' ')
        tag.insert_after('\n')
    for tag in soup.find_all('br'):
        tag.replace_with('\n')
    return soup.get_text('\n',strip=True)


def clean_markdown(value):
    value=html.unescape(value).replace('\r\n','\n').replace('\r','\n')
    images=re.findall(r'!\[[^\]]*\]\((https?://[^\s)]+)[^)]*\)',value)
    images += [urljoin('https://maplestar.io',tag.get('src')) for tag in BeautifulSoup(value,'html.parser').find_all('img') if tag.get('src')]
    text=re.sub(r'!\[[^\]]*\]\([^)]*\)','',value)
    if re.search(r'<(?:p|div|table|br|img|h[1-6])\b',text,re.I):
        text=html_text(text)
    return re.sub(r'\n{3,}','\n\n',re.sub(r'\n[ \t]+','\n',text)).strip(),list(dict.fromkeys(images))
