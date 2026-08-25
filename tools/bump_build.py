"""Bump the frontend build stamp everywhere it must agree: bump_build.py 2026-08-27a"""
import re, sys
b = sys.argv[1]
for p, pat, rep in (("frontend/app.js", r"APP_BUILD='[^']*'", f"APP_BUILD='{b}'"),
                    ("frontend/index.html", r'style\.css\?v=[^"]*', f"style.css?v={b}"),
                    ("frontend/index.html", r'app\.js\?v=[^"]*', f"app.js?v={b}")):
    s = open(p).read(); s2 = re.sub(pat, rep, s); open(p, "w").write(s2); print(p, "ok" if s2 != s else "(unchanged)")
