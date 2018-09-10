# ppoi(っぽい)

if文の条件式を明確に記述することはできないけど、良い例と悪い例は示せるってシチュエーションで、手軽に機械学習でそれっぽい判定をするのを支援するライブラリです。

メインスクリプトが __init__.py なので、チェックアウトしたディレクトリがそのままPythonのモジュールとしてインポート可能です。適当なわかりやすい名前にリネームすると良いです。

```
if keyword.ppoi(something):
   print("{} is keyword-ppoi".format(something))
```

__init__.py は直接実行可能です。 `--initialize` で実行すると良いです。

```
% __init__.py --initialize
```

使い方の説明 https://scrapbox.io/nishio/ppoi
