"""Build the revised GJI review files from their existing TeX masters."""
from scripts.export_ps_manuscript_markdown import run as markdown
from scripts.build_ps_supplement import run as supplement
from scripts.compile_ps_submission import run as compile_pdfs
from scripts.build_ps_chinese_review import run as chinese

if __name__ == '__main__':
    supplement()
    markdown()
    chinese()
    compile_pdfs()
