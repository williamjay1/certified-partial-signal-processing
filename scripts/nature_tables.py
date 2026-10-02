"""Publication tables from the manuscript's unchanged Markdown cells.

No training, prediction, statistical analysis, or numerical rounding is performed.
Physical export width is 183 mm; fonts remain editable in SVG and embedded in PDF.
"""
from pathlib import Path
import os, sys, json, hashlib, csv, re, math

ROOT = Path(__file__).resolve().parents[1]
for key, value in {
    'MPLCONFIGDIR': ROOT / 'cache' / 'matplotlib_nature_tables',
    'TEMP': ROOT / 'temp', 'TMP': ROOT / 'temp',
    'PYTHONPYCACHEPREFIX': ROOT / 'cache' / 'python_nature_tables',
}.items():
    os.environ[key] = str(value)
    Path(value).mkdir(parents=True, exist_ok=True)
for folder in (r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_packages',
               r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_core'):
    sys.path.insert(0, folder)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.backends.backend_agg import RendererAgg
from PIL import Image
from pypdf import PdfReader
import xml.etree.ElementTree as ET

SOURCE = ROOT / 'data' / 'table_cells.md'
OUT = ROOT / 'artwork' / 'tables'
OUT.mkdir(parents=True, exist_ok=True)
PT_MM = 72 / 25.4
WIDTH_MM = 183.0
WIDTH = WIDTH_MM * PT_MM
MARGIN = 4 * PT_MM
GRID = WIDTH - 2 * MARGIN
FS = 7.0
NOTE_FS = 6.5
COLOR = '#181818'
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': FS, 'text.color': COLOR,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'svg.hashsalt': 'ASC-NatureTables-v1',
})
FONT = findfont(FontProperties(family='Arial'), fallback_to_default=False)
RENDER_DPI = 300
MEASURER = RendererAgg(100, 100, RENDER_DPI)

SPECS = [
    dict(title='Related capabilities and the guarantee being evaluated', widths=[30,68,77],
         note='The present objective is pointwise relative multiclass half-Brier fidelity to the complete predictor.'),
    dict(title='Signal-processing workloads and frozen evaluation cohorts', widths=[20,30,47,32,46],
         note='HAR uses the official subject split. Hydraulic grouping uses operating-segment proxies, not documented independent runs.'),
    dict(title='Complete-model quality on the held-out HAR release', widths=[43,33,33,33,33],
         note='All 2,947 held-out windows; each system is a separately frozen training-subject realization. Accuracy is a percentage.'),
    dict(title='Complete-model processing cost and same-cohort quality', widths=[43,38,37,29,28],
         note='Same 900 windows, three systems and three randomized repetitions per implementation. Complete time includes all 18 raw-signal blocks; this is a separate benchmark from the primary partial/full experiment.'),
    dict(title='Primary HAR comparison at half-Brier tolerance 0.025', widths=[14,24,24,23,39,25,26], bold_rows=[3],
         note='Same 900 windows per system. Complete time includes feature processing, enclosure, certification, control and exact fallback. Mean is the equal-system pooled summary; blocks and fallback refer to partial processing.'),
    dict(title='HAR stopping and output-construction controls', widths=[49,26,21,23,28,28],
         note='Same 900 windows for each of three systems; 2,700 system–windows in the exceedance denominator. Mean observed-label increase and every-label exceedance are distinct criteria.'),
    dict(title='Prescribed fidelity tolerance and measured complete time', widths=[20,26,20,31.5,26,20,31.5],
         note='HAR uses the same 900-window timing cohort across all three systems. Hydraulic uses all 435 confirmation cycles. Reductions are against contemporaneous compiled full execution.',
         groups=[(1,3,'Smartphone HAR'),(4,6,'Hydraulic confirmation')],
         display_headers=['Half-Brier tolerance','Complete time, ms','Reduction','Same-cohort mean increase','Complete time, ms','Reduction','Mean increase']),
    dict(title='Complete and partial latency quantiles', widths=[49,42,42,42],
         note='Each pair is full / partial. HAR quantiles use each system’s 2,700 recorded calls; Hydraulic uses 1,305 calls. Quantiles describe measured latency, not hard real-time bounds.'),
    dict(title='Primary HAR latency conditioned on processing outcome', widths=[38,27,46,39,25],
         note='2,700 system–window pairs; each mean first averages three repetitions for the same system–window. Partial slower is the fraction whose paired mean is larger, not a causal effect of fallback.'),
    dict(title='Hydraulic confirmation at half-Brier tolerance 0.025', widths=[50,27,29,23,22,24],
         note='435 frozen confirmation cycles; complete time includes all online work. Mean blocks are out of 28. The coordinate certificate is cheaper here despite its looser relaxation.'),
    dict(title='Development-only sensitivity to feature-block order', widths=[50,24,24,38.5,38.5],
         note='Equal-system means on the original development subjects; no new raw-signal timing or test-policy selection. Diameter upper proxy is computed per case before averaging and does not measure the exact enclosure gap.'),
    dict(title='Requirements for extending the partial enclosure', widths=[33,65,77],
         note='These routes are proposed extensions. Only numerical multiclass HGB trees with at most 63 leaves per tree were implemented and evaluated.'),
]

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def cells(line):
    return [x.strip() for x in line.strip().strip('|').split('|')]

def extract_tables(text):
    lines = text.splitlines()
    tables = []
    idx = 0
    while idx < len(lines):
        if lines[idx].lstrip().startswith('|'):
            start = idx
            block = []
            while idx < len(lines) and lines[idx].lstrip().startswith('|'):
                block.append(lines[idx])
                idx += 1
            if len(block) < 3 or not all(re.fullmatch(r':?-{3,}:?', c) for c in cells(block[1])):
                raise RuntimeError(f'Not a well-formed Markdown table at line {start + 1}')
            header = cells(block[0])
            rows = [cells(x) for x in block[2:]]
            assert all(len(r) == len(header) for r in rows)
            align = ['right' if x.endswith(':') else 'left' for x in cells(block[1])]
            # Label columns are always left-aligned; numeric tolerance remains right-aligned.
            if not all(re.fullmatch(r'[\d.]+', r[0]) for r in rows):
                align[0] = 'left'
            tables.append(dict(line_start=start + 1, line_end=idx, header=header,
                               rows=rows, align=align, markdown='\n'.join(block)))
        else:
            idx += 1
    return tables

def text_width_pt(text, font_size=FS, bold=False):
    prop = FontProperties(family='Arial', size=font_size, weight='bold' if bold else 'normal')
    return MEASURER.get_text_width_height_descent(text, prop, False)[0] * 72 / RENDER_DPI

def wrap(text, max_width, font_size=FS, bold=False):
    words = text.split()
    result, current = [], ''
    for word in words:
        candidate = f'{current} {word}'.strip()
        if current and text_width_pt(candidate, font_size, bold) > max_width:
            result.append(current)
            current = word
        else:
            current = candidate
        if text_width_pt(current, font_size, bold) > max_width + 0.05:
            raise ValueError(f'Unbreakable word exceeds column: {current!r}')
    if current:
        result.append(current)
    return '\n'.join(result)

def text_height(wrapped, font_size=FS):
    # Deliberately larger than glyph extents, leaving whitespace around horizontal rules.
    return max(1, wrapped.count('\n') + 1) * font_size * 1.35

def typography(text):
    # Display spacing only; CSV/Markdown and canonical cell hashes retain the original bytes.
    text=re.sub(r';(?=\S)', '; ', text)
    return re.sub(r'(?<=\d)/(?=\d)', ' / ', text)

def positive_overlap(a, b, tolerance=0.1):
    return min(a.x1,b.x1) - max(a.x0,b.x0) > tolerance and min(a.y1,b.y1) - max(a.y0,b.y0) > tolerance

def make_table(table, spec, number):
    widths = [x * PT_MM for x in spec['widths']]
    assert abs(sum(widths) - GRID) < 1e-6
    pad = 3.5
    header = spec.get('display_headers', table['header'])
    wh = [wrap(c, w-2*pad, FS, True) for c,w in zip(header,widths)]
    wr = [[wrap(typography(c), w-2*pad, FS, i in spec.get('bold_rows', [])) for c,w in zip(row,widths)]
          for i,row in enumerate(table['rows'])]
    group_height = 18 if spec.get('groups') else 0
    head_height = max(map(text_height,wh)) + 12
    row_heights = [max(map(text_height,row)) + 12 for row in wr]
    title = f'Table {number} | {spec["title"]}'
    title_wrap = wrap(title, GRID, 8, True)
    title_height = text_height(title_wrap,8)
    note = wrap(spec['note'], GRID, NOTE_FS)
    note_height = text_height(note,NOTE_FS)
    top_pad = 10
    after_title = 10
    note_gap = 9
    bottom_pad = 11
    height = top_pad + title_height + after_title + group_height + head_height + sum(row_heights) + note_gap + note_height + bottom_pad
    if height / PT_MM > 170:
        raise ValueError(f'Table {number} exceeds 170 mm maximum height')
    fig = plt.figure(figsize=(WIDTH/72,height/72), dpi=RENDER_DPI)
    ax = fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,WIDTH),ylim=(0,height))
    ax.axis('off')
    texts, rules, cells_meta = [], [], []

    def put(text,x,top,fontsize=FS,bold=False,ha='left',kind='cell', bounds=None):
        obj = ax.text(x,height-top,text,fontsize=fontsize,
                      fontweight='bold' if bold else 'normal',ha=ha,va='top',
                      linespacing=1.35,clip_on=False,color=COLOR)
        texts.append((obj,kind,bounds))
        return obj

    def rule(top,weight=0.5,x0=MARGIN,x1=WIDTH-MARGIN):
        line = ax.plot([x0,x1],[height-top,height-top],color=COLOR,lw=weight,
                       solid_capstyle='butt',clip_on=False)[0]
        rules.append((line,top,x0,x1,weight))

    put(title_wrap,MARGIN,top_pad,8,True,kind='title')
    top = top_pad + title_height + after_title
    rule(top,0.8)
    colx = [MARGIN]
    for w in widths:
        colx.append(colx[-1]+w)
    if spec.get('groups'):
        for first,last,label in spec['groups']:
            put(label,(colx[first]+colx[last+1])/2,top+5,FS,True,'center','group_header')
        top += group_height
        for first,last,_ in spec['groups']:
            rule(top,0.4,colx[first]+pad,colx[last+1]-pad)
    for j,c in enumerate(wh):
        align = table['align'][j]
        x = colx[j]+pad if align == 'left' else colx[j+1]-pad
        bounds = (colx[j],top,colx[j+1],top+head_height)
        put(c,x,top+6,FS,True,align,'column_header',bounds)
        cells_meta.append({'row':'header','column':j,'original':table['header'][j],
                           'display':c,'cell_bounds_pt':bounds})
    top += head_height
    rule(top,0.5)
    for i,(row,row_height) in enumerate(zip(wr,row_heights)):
        for j,c in enumerate(row):
            align=table['align'][j]
            x=colx[j]+pad if align=='left' else colx[j+1]-pad
            # Descriptive tables are top-aligned; numeric tables align cell centers vertically.
            offset=6 if len(widths)==3 else 6+(row_height-12-text_height(c))/2
            bounds=(colx[j],top,colx[j+1],top+row_height)
            put(c,x,top+offset,FS,i in spec.get('bold_rows',[]),align,'body_cell',bounds)
            cells_meta.append({'row':i,'column':j,'original':table['rows'][i][j],
                               'display':c,'cell_bounds_pt':bounds})
        top += row_height
    rule(top,0.8)
    put(note,MARGIN,top+note_gap,NOTE_FS,kind='note')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    scale=RENDER_DPI/72
    bboxes=[obj.get_window_extent(renderer) for obj,_,_ in texts]
    collisions=[]
    line_clearances=[]
    for i,a in enumerate(bboxes):
        for j,b in enumerate(bboxes[:i]):
            if positive_overlap(a,b,0.15*scale):
                collisions.append({'type':'text_text','a':i,'b':j,
                                   'text_a':texts[i][0].get_text(),'text_b':texts[j][0].get_text()})
        # Including half the visible rule stroke, preserve at least 1.5 pt clearance.
        for rule_idx,(_,line_top,x0,x1,weight) in enumerate(rules):
            line_y=(height-line_top)*scale
            intersects_x=min(a.x1,x1*scale)>max(a.x0,x0*scale)
            clearance=min(abs(a.y0-line_y),abs(a.y1-line_y))/scale-weight/2
            if intersects_x and a.y0 <= line_y <= a.y1:
                clearance=-1
            if intersects_x:
                line_clearances.append(clearance)
            if intersects_x and clearance < 1.5:
                collisions.append({'type':'text_rule','text':i,'rule':rule_idx,'clearance_pt':clearance})
        if a.x0 < -0.1 or a.y0 < -0.1 or a.x1 > WIDTH*scale+0.1 or a.y1 > height*scale+0.1:
            collisions.append({'type':'canvas_clipping','text':i})
        obj,kind,bounds=texts[i]
        if bounds:
            x0,t0,x1,t1=bounds
            if a.x0 < x0*scale-0.1 or a.x1 > x1*scale+0.1 or a.y1 > (height-t0)*scale+0.1 or a.y0 < (height-t1)*scale-0.1:
                collisions.append({'type':'cell_overflow','text':i,'kind':kind})
    if collisions:
        plt.close(fig)
        raise ValueError(json.dumps({'table':number,'collisions':collisions},ensure_ascii=False))
    name=f'table{number:02d}'
    outputs={}
    metadata={'Title':title,'Author':'','Subject':'Unchanged manuscript data; no reanalysis','Keywords':'publication table'}
    for ext in ('svg','pdf','png'):
        path=OUT/f'{name}.{ext}'
        fig.savefig(path,dpi=1000,bbox_inches=None,pad_inches=0,metadata=metadata if ext=='pdf' else None)
        outputs[ext]={'path':str(path),'sha256':sha_bytes(path.read_bytes())}
    preview=OUT/f'{name}_preview.png'
    fig.savefig(preview,dpi=300,bbox_inches=None,pad_inches=0)
    plt.close(fig)
    csv_path=OUT/f'{name}_source.csv'
    with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(table['header']);writer.writerows(table['rows'])
    (OUT/f'{name}_source.md').write_text(table['markdown']+'\n',encoding='utf-8')
    canonical=json.dumps([table['header'],*table['rows']],ensure_ascii=False,separators=(',',':')).encode('utf-8')
    # Wrapping and grouped headings must never alter any body-cell content.
    assert all(re.sub(r'\s+','',cell['display'])==re.sub(r'\s+','',cell['original'])
               for cell in cells_meta if cell['row']!='header')
    with Image.open(OUT/f'{name}.png') as image:
        raster={'pixels':list(image.size),'dpi_metadata':list(image.info.get('dpi',[]))}
    svg=(OUT/f'{name}.svg').read_text(encoding='utf-8')
    svg_text_count=svg.count('<text')
    if svg_text_count<len(texts):
        raise ValueError(f'SVG editable text is missing for table {number}')
    svg_root=ET.fromstring(svg)
    svg_images=svg_root.findall('.//{http://www.w3.org/2000/svg}image')
    reader=PdfReader(OUT/f'{name}.pdf')
    page=reader.pages[0]
    assert len(reader.pages)==1 and len(page.images)==0 and not svg_images
    pdf_fonts=[]
    for _,ref in page['/Resources']['/Font'].items():
        font=ref.get_object()
        descendant=font['/DescendantFonts'][0].get_object() if '/DescendantFonts' in font else font
        descriptor=descendant.get('/FontDescriptor')
        embedded=descriptor is not None and '/FontFile2' in descriptor.get_object()
        assert embedded, f'PDF font not embedded: {font}'
        pdf_fonts.append({'base_font':str(font.get('/BaseFont')),'subtype':str(font.get('/Subtype')),
                          'descendant_subtype':str(descendant.get('/Subtype')),'TrueType_embedded':embedded})
    return {
        'table':number,'title':spec['title'],'source_line_start':table['line_start'],
        'source_line_end':table['line_end'],'source_cells_sha256':sha_bytes(canonical),
        'body_rows':len(table['rows']),'columns':len(table['header']),
        'width_mm':WIDTH_MM,'height_mm':height/PT_MM,'body_font_pt':FS,
        'minimum_font_pt':NOTE_FS,'font_file':FONT,'column_width_mm':spec['widths'],
        'row_heights_pt':row_heights,'collision_checks':{
            'text_text':'PASS','text_horizontal_rule':'PASS','cell_overflow':'PASS',
            'canvas_clipping':'PASS','required_text_rule_clearance_pt':1.5,
            'measured_minimum_text_rule_clearance_pt':min(line_clearances),'text_objects':len(texts)},
        'unchanged_body_content_except_display_whitespace':'PASS','svg_editable_text_count':svg_text_count,
        'vector_verification':{'svg_embedded_images':len(svg_images),'pdf_embedded_images':len(page.images),
                               'pdf_fonts':pdf_fonts,'pdf_width_mm':float(page.mediabox.width)/PT_MM,
                               'pdf_height_mm':float(page.mediabox.height)/PT_MM},
        'raster':raster,'outputs':outputs,'preview':str(preview),'cells':cells_meta,
    }

def main():
    text=SOURCE.read_text(encoding='utf-8')
    tables=extract_tables(text)
    if len(tables)!=len(SPECS):
        raise ValueError(f'Expected {len(SPECS)} tables, found {len(tables)}')
    results=[]
    for number,(table,spec) in enumerate(zip(tables,SPECS),1):
        result=make_table(table,spec,number)
        results.append(result)
        print(f'Table {number:02d}: {result["body_rows"]} rows × {result["columns"]} columns; '
              f'{result["width_mm"]:.1f} × {result["height_mm"]:.1f} mm; collision checks PASS',flush=True)
    manifest={'schema':'ASC-publication-tables-v1','source':str(SOURCE),
              'source_sha256':sha_bytes(SOURCE.read_bytes()),
              'script_sha256':sha_bytes(Path(__file__).read_bytes()),
              'format':'183-mm Nature-style three-rule tables; no vertical lines or colored fills',
              'export_png_dpi':1000,'preview_png_dpi':300,'tables':results}
    (OUT/'table_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'COMPLETE: {len(results)} tables; all cells retained; editable SVG/PDF + 1000-dpi PNG.',flush=True)

if __name__=='__main__':
    main()
