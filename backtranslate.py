"""
    To download more, visit: https://www.argosopentech.com/argospm/index/
    Always download and install both ways!
"""

from argostranslate import package, translate

# English 2 Spanish 2 English
package.install_from_path('argostranslate/translate-en_es-1_0.argosmodel')
package.install_from_path('argostranslate/translate-es_en-1_0.argosmodel')

# English 2 Arabic 2 English
package.install_from_path('argostranslate/translate-en_ar-1_0.argosmodel')
package.install_from_path('argostranslate/translate-ar_en-1_0.argosmodel')

installed_languages = translate.get_installed_languages()

def backtranslation(s, i): 
    # s stands for string, i stands for indices (list)
    # 0 is English, 1 is Arabic - implement way to find it TODO
    forward = installed_languages[i[0]].get_translation(installed_languages[i[1]])
    back = installed_languages[i[1]].get_translation(installed_languages[i[0]])
    return back.translate(forward.translate(s))


if __name__ == "__main__":
    english = "Hello World!"
    print(backtranslation(english, [0, 2]))
