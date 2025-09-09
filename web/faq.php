<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>よくある質問 - Chord.codes</title>
<script src="https://cdn.tailwindcss.com/3.4.16"></script>
<script>
tailwind.config = {
theme: {
extend: {
colors: {
primary: '#2E1B69',
secondary: '#1BB6B6'
},
borderRadius: {
'none': '0px',
'sm': '4px',
DEFAULT: '8px',
'md': '12px',
'lg': '16px',
'xl': '20px',
'2xl': '24px',
'3xl': '32px',
'full': '9999px',
'button': '8px'
}
}
}
}
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Pacifico&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/remixicon/4.6.0/remixicon.min.css" rel="stylesheet">
<style>
:where([class^="ri-"])::before {
content: "\f3c2";
}
.gradient-bg {
background: linear-gradient(135deg, #2E1B69 0%, #1B69B6 50%, #1BB6B6 100%);
}
.card-glow {
box-shadow: 0 0 30px rgba(27, 182, 182, 0.1);
}
.text-gradient {
background: linear-gradient(135deg, #1BB6B6, #1B69B6);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
}
.tab-active {
background: linear-gradient(135deg, #2E1B69, #1BB6B6);
}
.faq-item {
transition: all 0.3s ease;
}
.faq-item:hover {
transform: translateY(-2px);
box-shadow: 0 8px 25px rgba(27, 182, 182, 0.15);
}
.faq-answer {
max-height: 0;
overflow: hidden;
transition: max-height 0.3s ease;
}
.faq-answer.active {
max-height: 500px;
}
.category-tab {
transition: all 0.3s ease;
}
.category-tab:hover {
background-color: rgba(27, 182, 182, 0.1);
}
</style>
</head>
<body class="bg-gray-50 text-gray-900">
<header class="bg-white shadow-sm border-b border-gray-200">
<nav class="flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
<div class="flex items-center space-x-3">
<div class="w-10 h-10 flex items-center justify-center bg-primary rounded-lg">
<i class="ri-music-2-fill text-white text-xl"></i>
</div>
<span class="font-['Pacifico'] text-2xl text-primary">Chord.codes</span>
</div>
<div class="flex items-center space-x-6">
<a href="/" data-readdy="true" class="flex items-center space-x-2 text-gray-600 hover:text-primary transition-colors">
<i class="ri-arrow-left-line"></i>
<span>ホームに戻る</span>
</a>
<button class="md:hidden w-8 h-8 flex items-center justify-center text-gray-600">
<i class="ri-menu-line text-xl"></i>
</button>
</div>
</nav>
</header>
<main>
<section class="py-16 bg-white">
<div class="max-w-6xl mx-auto px-8">
<div class="text-center mb-12">
<h1 class="text-4xl md:text-5xl font-bold mb-6">
<span class="text-gradient">よくある質問</span>
</h1>
<p class="text-xl text-gray-600 leading-relaxed max-w-3xl mx-auto">
Chord.codes に関するよくある質問と回答をまとめました。お探しの情報が見つからない場合は、お気軽にサポートチームまでお問い合わせください。
</p>
</div>
<div class="mb-12">
<div class="relative max-w-2xl mx-auto">
<input type="text" id="search-input" placeholder="キーワードで検索..." class="w-full px-6 py-4 pl-14 text-lg border border-gray-200 !rounded-button focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-white">
<div class="absolute left-4 top-1/2 transform -translate-y-1/2 w-6 h-6 flex items-center justify-center">
<i class="ri-search-line text-gray-400 text-xl"></i>
</div>
</div>
</div>
<div class="mb-8">
<div class="flex space-x-2 bg-gray-100 p-1 rounded-lg overflow-x-auto">
<button class="category-tab px-6 py-2 !rounded-button font-medium transition-all tab-active text-white whitespace-nowrap" data-category="all">すべて</button>
<button class="category-tab px-6 py-2 !rounded-button font-medium transition-all text-gray-600 hover:text-gray-900 whitespace-nowrap" data-category="usage">アプリの使い方</button>
<button class="category-tab px-6 py-2 !rounded-button font-medium transition-all text-gray-600 hover:text-gray-900 whitespace-nowrap" data-category="system">システム要件</button>
<button class="category-tab px-6 py-2 !rounded-button font-medium transition-all text-gray-600 hover:text-gray-900 whitespace-nowrap" data-category="midi">MIDI 接続</button>
<button class="category-tab px-6 py-2 !rounded-button font-medium transition-all text-gray-600 hover:text-gray-900 whitespace-nowrap" data-category="pricing">価格・支払い</button>
<button class="category-tab px-6 py-2 !rounded-button font-medium transition-all text-gray-600 hover:text-gray-900 whitespace-nowrap" data-category="troubleshooting">トラブルシューティング</button>
</div>
</div>
<div id="faq-container" class="space-y-4">
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="usage">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">Chord.codes はどのような音楽アプリですか？</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p class="mb-4">Chord.codes は AI 技術を活用したリアルタイム音楽生成アプリです。お手持ちの電子ピアノや MIDI キーボードと接続することで、AI が無限に音楽を生成し、自動演奏を楽しむことができます。</p>
<p>従来の録音された楽曲とは異なり、毎回新しい音楽が生成されるため、同じ演奏は二度と聞くことができません。クラシック、ジャズ、ポップスなど様々な音楽スタイルに対応しており、あなたの好みに合わせてカスタマイズも可能です。</p>
</div>
</div>
</div>
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="usage">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">アプリの基本的な使い方を教えてください</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p class="mb-4">基本的な使用手順は以下の通りです：</p>
<ol class="list-decimal list-inside space-y-2 mb-4">
<li>アプリをダウンロードしてインストール</li>
<li>電子ピアノを USB-C ケーブルでスマートフォンに接続</li>
<li>アプリを起動し、MIDI 機器が自動検出されることを確認</li>
<li>お好みの音楽スタイルを選択</li>
<li>「演奏開始」ボタンをタップして AI 音楽生成を開始</li>
</ol>
<p>初回起動時にはチュートリアルが表示されるため、初心者の方でも安心してご利用いただけます。</p>
</div>
</div>
</div>
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="system">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">対応しているスマートフォンの機種を教えてください</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p class="mb-4">以下の要件を満たすデバイスでご利用いただけます：</p>
<div class="grid md:grid-cols-2 gap-6">
<div>
<h4 class="font-semibold mb-2 text-primary">iOS デバイス</h4>
<ul class="list-disc list-inside space-y-1">
<li>iOS 14.0 以降</li>
<li>iPhone 8 以降のモデル</li>
<li>128 MB 以上の空き容量</li>
<li>USB-C 対応（iPhone 15 以降推奨）</li>
</ul>
</div>
<div>
<h4 class="font-semibold mb-2 text-primary">Android デバイス</h4>
<ul class="list-disc list-inside space-y-1">
<li>Android 8.0 以降</li>
<li>RAM 4 GB 以上推奨</li>
<li>128 MB 以上の空き容量</li>
<li>USB-C ポート必須</li>
</ul>
</div>
</div>
</div>
</div>
</div>
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="system">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">インターネット接続は必要ですか？</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p class="mb-4">はい、Chord.codes の AI 音楽生成機能を使用するには安定したインターネット接続が必要です。</p>
<p class="mb-4">推奨される接続環境：</p>
<ul class="list-disc list-inside space-y-1 mb-4">
<li>Wi-Fi 接続（推奨）</li>
<li>4G/5G モバイルデータ通信</li>
<li>最低 1 Mbps の通信速度</li>
</ul>
<p>AI による音楽生成処理はクラウド上で行われるため、リアルタイムでの通信が必要となります。通信環境が不安定な場合、音楽生成に遅延が生じる可能性があります。</p>
</div>
</div>
</div>
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="midi">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">どのような電子ピアノ・キーボードが対応していますか？</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p class="mb-4">USB-C ポートを搭載した MIDI 対応の電子ピアノ・キーボードであれば、ほとんどの機種でご利用いただけます。</p>
<div class="grid md:grid-cols-2 gap-6 mb-4">
<div>
<h4 class="font-semibold mb-2 text-primary">推奨機種</h4>
<ul class="list-disc list-inside space-y-1 text-sm">
<li>Yamaha P-125, P-515</li>
<li>Roland FP-30X, FP-60X</li>
<li>Casio PX-S1100, PX-S3100</li>
<li>Korg B2SP, LP-380</li>
<li>Kawai ES120, ES920</li>
<li>Nord Piano 5, Stage 4</li>
</ul>
</div>
<div>
<h4 class="font-semibold mb-2 text-primary">必要な条件</h4>
<ul class="list-disc list-inside space-y-1 text-sm">
<li>USB-C ポート搭載</li>
<li>MIDI 出力対応</li>
<li>88 鍵盤推奨（61 鍵盤でも可）</li>
<li>USB バスパワー対応</li>
</ul>
</div>
</div>
<p>古い機種で USB-A ポートのみの場合は、USB-A to USB-C 変換アダプターをご利用ください。</p>
</div>
</div>
</div>
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="midi">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">MIDI 機器が認識されない場合の対処法は？</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p class="mb-4">以下の手順をお試しください：</p>
<ol class="list-decimal list-inside space-y-2 mb-4">
<li><strong>ケーブル接続の確認</strong><br>USB-C ケーブルがしっかりと接続されているか確認してください</li>
<li><strong>電子ピアノの電源確認</strong><br>MIDI 機器の電源が入っており、MIDI 出力が有効になっているか確認</li>
<li><strong>ケーブルの種類確認</strong><br>データ転送対応の USB-C ケーブルを使用してください（充電専用ケーブルは不可）</li>
<li><strong>アプリの再起動</strong><br>Chord.codes アプリを一度終了し、再起動してください</li>
<li><strong>デバイスの再起動</strong><br>スマートフォンを再起動してから再度接続をお試しください</li>
</ol>
<p>それでも解決しない場合は、お使いの機種名とともにサポートまでお問い合わせください。</p>
</div>
</div>
</div>
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="pricing">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">無料版とプレミアム版の違いは何ですか？</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<div class="grid md:grid-cols-2 gap-6">
<div class="bg-gray-50 rounded-lg p-4">
<h4 class="font-semibold mb-3 text-gray-900">無料版</h4>
<ul class="list-disc list-inside space-y-1 text-sm">
<li>基本的な AI 音楽生成</li>
<li>3 つの音楽スタイル</li>
<li>1 時間連続演奏</li>
<li>広告表示あり</li>
</ul>
</div>
<div class="bg-gradient-to-br from-primary to-secondary rounded-lg p-4 text-white">
<h4 class="font-semibold mb-3">プレミアム版（月額 ¥980）</h4>
<ul class="list-disc list-inside space-y-1 text-sm">
<li>無制限 AI 音楽生成</li>
<li>20+ 音楽スタイル</li>
<li>24 時間連続演奏</li>
<li>楽曲保存・共有機能</li>
<li>カスタムスタイル作成</li>
<li>広告なし</li>
</ul>
</div>
</div>
<p class="mt-4">プレミアム版は 7 日間無料でお試しいただけます。期間中はいつでもキャンセル可能です。</p>
</div>
</div>
</div>
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="pricing">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">支払い方法と解約について教えてください</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p class="mb-4"><strong>支払い方法：</strong></p>
<ul class="list-disc list-inside space-y-1 mb-4">
<li>App Store 経由（iOS）：Apple ID に登録されたクレジットカード、デビットカード、Apple Pay</li>
<li>Google Play 経由（Android）：Google アカウントに登録された支払い方法</li>
</ul>
<p class="mb-4"><strong>解約方法：</strong></p>
<ul class="list-disc list-inside space-y-1 mb-4">
<li>iOS：設定 > Apple ID > サブスクリプション > Chord.codes > サブスクリプションをキャンセル</li>
<li>Android：Google Play ストア > メニュー > 定期購入 > Chord.codes > 解約</li>
</ul>
<p>解約後も現在の課金期間終了まではプレミアム機能をご利用いただけます。</p>
</div>
</div>
</div>
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="troubleshooting">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">音楽生成が遅い、または止まってしまいます</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p class="mb-4">以下の原因と対処法をご確認ください：</p>
<div class="space-y-4">
<div>
<h4 class="font-semibold text-gray-900 mb-2">ネットワーク環境の問題</h4>
<ul class="list-disc list-inside space-y-1 text-sm">
<li>Wi-Fi 接続を確認し、可能であれば 5GHz 帯域に接続</li>
<li>モバイルデータ使用時は電波状況の良い場所に移動</li>
<li>他のアプリでのデータ使用を一時停止</li>
</ul>
</div>
<div>
<h4 class="font-semibold text-gray-900 mb-2">デバイスの負荷</h4>
<ul class="list-disc list-inside space-y-1 text-sm">
<li>バックグラウンドで動作している不要なアプリを終了</li>
<li>デバイスの再起動を行う</li>
<li>ストレージ容量に十分な空きがあることを確認</li>
</ul>
</div>
<div>
<h4 class="font-semibold text-gray-900 mb-2">アプリの設定</h4>
<ul class="list-disc list-inside space-y-1 text-sm">
<li>音質設定を「標準」に変更してお試しください</li>
<li>アプリを最新版にアップデート</li>
</ul>
</div>
</div>
</div>
</div>
</div>
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="troubleshooting">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">アプリがクラッシュ・強制終了してしまいます</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p class="mb-4">以下の手順で問題の解決をお試しください：</p>
<ol class="list-decimal list-inside space-y-2 mb-4">
<li><strong>アプリの完全終了と再起動</strong><br>アプリを完全に終了し、再度起動してください</li>
<li><strong>デバイスの再起動</strong><br>スマートフォンを再起動してから再度お試しください</li>
<li><strong>アプリの再インストール</strong><br>アプリを削除し、App Store または Google Play から再インストール</li>
<li><strong>OS のアップデート確認</strong><br>デバイスの OS が最新版であることを確認</li>
<li><strong>ストレージ容量の確認</strong><br>デバイスに十分な空き容量があることを確認</li>
</ol>
<p>問題が継続する場合は、お使いのデバイス情報（機種名、OS バージョン）とともにサポートまでご連絡ください。</p>
</div>
</div>
</div>
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="usage">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">生成された音楽を保存・共有することはできますか？</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p class="mb-4">楽曲の保存・共有機能はプレミアム版でご利用いただけます。</p>
<div class="space-y-4">
<div>
<h4 class="font-semibold text-gray-900 mb-2">保存機能</h4>
<ul class="list-disc list-inside space-y-1 text-sm">
<li>気に入った楽曲をライブラリに保存</li>
<li>最大 100 曲まで保存可能</li>
<li>MIDI ファイル形式でのエクスポート</li>
<li>音声ファイル（MP3）としての書き出し</li>
</ul>
</div>
<div>
<h4 class="font-semibold text-gray-900 mb-2">共有機能</h4>
<ul class="list-disc list-inside space-y-1 text-sm">
<li>SNS への直接投稿</li>
<li>メール・メッセージでの共有</li>
<li>Chord.codes コミュニティでの公開</li>
<li>QR コードでの楽曲共有</li>
</ul>
</div>
</div>
<p class="mt-4">無料版では楽曲の一時的な再生のみ可能で、保存機能はご利用いただけません。</p>
</div>
</div>
</div>
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 card-glow cursor-pointer" data-category="troubleshooting">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">音が出ない、または音質が悪い場合の対処法は？</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p class="mb-4">音響に関する問題の解決方法をご案内します：</p>
<div class="space-y-4">
<div>
<h4 class="font-semibold text-gray-900 mb-2">音が出ない場合</h4>
<ul class="list-disc list-inside space-y-1 text-sm">
<li>電子ピアノの音量設定を確認</li>
<li>スマートフォンの音量設定を確認</li>
<li>ヘッドフォン・スピーカーの接続状況を確認</li>
<li>MIDI 接続が正常に行われているか確認</li>
<li>アプリ内の音声出力設定を確認</li>
</ul>
</div>
<div>
<h4 class="font-semibold text-gray-900 mb-2">音質が悪い場合</h4>
<ul class="list-disc list-inside space-y-1 text-sm">
<li>アプリ内の音質設定を「高品質」に変更</li>
<li>ネットワーク接続を安定した Wi-Fi に切り替え</li>
<li>USB-C ケーブルの品質を確認（高品質なケーブルを使用）</li>
<li>電子ピアノの内蔵エフェクトをオフに設定</li>
<li>他のアプリからの音声出力を停止</li>
</ul>
</div>
</div>
<p class="mt-4">問題が解決しない場合は、お使いの機器構成とともにサポートまでお問い合わせください。</p>
</div>
</div>
</div>
</div>
<div class="text-center mt-12">
<div class="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-2xl p-8">
<h3 class="text-2xl font-bold mb-4">お探しの情報が見つかりませんか？</h3>
<p class="text-gray-600 mb-6">
サポートチームが丁寧にお答えいたします。お気軽にお問い合わせください。
</p>
<div class="flex flex-col sm:flex-row gap-4 justify-center">
<a href="#" class="flex items-center justify-center space-x-2 bg-primary text-white px-6 py-3 !rounded-button hover:bg-opacity-90 transition-colors whitespace-nowrap">
<div class="w-5 h-5 flex items-center justify-center">
<i class="ri-mail-line"></i>
</div>
<span>メールでお問い合わせ</span>
</a>
<a href="#" class="flex items-center justify-center space-x-2 bg-secondary text-white px-6 py-3 !rounded-button hover:bg-opacity-90 transition-colors whitespace-nowrap">
<div class="w-5 h-5 flex items-center justify-center">
<i class="ri-chat-3-line"></i>
</div>
<span>チャットサポート</span>
</a>
</div>
</div>
</div>
</div>
</section>
</main>
<footer class="bg-gray-900 text-white py-12">
<div class="max-w-6xl mx-auto px-8">
<div class="grid md:grid-cols-4 gap-8">
<div class="md:col-span-2">
<div class="flex items-center space-x-3 mb-6">
<div class="w-10 h-10 flex items-center justify-center bg-white rounded-lg">
<i class="ri-music-2-fill text-primary text-xl"></i>
</div>
<span class="font-['Pacifico'] text-2xl text-white">Chord.codes</span>
</div>
<p class="text-gray-400 leading-relaxed mb-6">
AI が生成する無限の音楽をリアルタイムで演奏する革新的なアプリケーション。生演奏の魅力を現代のテクノロジーで再定義します。
</p>
<div class="flex space-x-4">
<a href="#" class="w-10 h-10 flex items-center justify-center bg-gray-800 rounded-lg hover:bg-secondary transition-colors">
<i class="ri-twitter-x-line text-white"></i>
</a>
<a href="#" class="w-10 h-10 flex items-center justify-center bg-gray-800 rounded-lg hover:bg-secondary transition-colors">
<i class="ri-youtube-line text-white"></i>
</a>
<a href="#" class="w-10 h-10 flex items-center justify-center bg-gray-800 rounded-lg hover:bg-secondary transition-colors">
<i class="ri-github-line text-white"></i>
</a>
</div>
</div>
<div>
<h3 class="font-semibold text-white mb-4">製品</h3>
<ul class="space-y-2 text-gray-400">
<li><a href="#" class="hover:text-white transition-colors">機能紹介</a></li>
<li><a href="#" class="hover:text-white transition-colors">技術仕様</a></li>
<li><a href="#" class="hover:text-white transition-colors">価格</a></li>
<li><a href="faq.php" class="hover:text-white transition-colors">FAQ</a></li>
</ul>
</div>
<div>
<h3 class="font-semibold text-white mb-4">サポート</h3>
<ul class="space-y-2 text-gray-400">
<li><a href="support.php" data-readdy="true" class="hover:text-white transition-colors">お問い合わせ</a></li>
<li><a href="privacy.php" class="hover:text-white transition-colors">プライバシーポリシー</a></li>
<li><a href="terms.php" class="hover:text-white transition-colors">利用規約</a></li>
</ul>
</div>
</div>
<div class="border-t border-gray-800 mt-12 pt-8 text-center text-gray-400">
<p>&copy; 2025 Chord.codes. All rights reserved.</p>
</div>
</div>
</footer>
<script id="faq-functionality">
document.addEventListener('DOMContentLoaded', function() {
const faqItems = document.querySelectorAll('.faq-item');
const categoryTabs = document.querySelectorAll('.category-tab');
const searchInput = document.getElementById('search-input');
faqItems.forEach(item => {
item.addEventListener('click', function() {
const answer = this.querySelector('.faq-answer');
const icon = this.querySelector('.faq-icon');
const isActive = answer.classList.contains('active');
faqItems.forEach(otherItem => {
const otherAnswer = otherItem.querySelector('.faq-answer');
const otherIcon = otherItem.querySelector('.faq-icon');
otherAnswer.classList.remove('active');
otherIcon.classList.remove('ri-subtract-line');
otherIcon.classList.add('ri-add-line');
});
if (!isActive) {
answer.classList.add('active');
icon.classList.remove('ri-add-line');
icon.classList.add('ri-subtract-line');
}
});
});
categoryTabs.forEach(tab => {
tab.addEventListener('click', function() {
const category = this.dataset.category;
categoryTabs.forEach(t => {
t.classList.remove('tab-active', 'text-white');
t.classList.add('text-gray-600');
});
this.classList.add('tab-active', 'text-white');
this.classList.remove('text-gray-600');
faqItems.forEach(item => {
if (category === 'all' || item.dataset.category === category) {
item.style.display = 'block';
} else {
item.style.display = 'none';
}
});
});
});
searchInput.addEventListener('input', function() {
const searchTerm = this.value.toLowerCase();
let hasResults = false;
faqItems.forEach(item => {
const question = item.querySelector('h3').textContent.toLowerCase();
const answer = item.querySelector('.faq-answer').textContent.toLowerCase();
if (question.includes(searchTerm) || answer.includes(searchTerm)) {
item.style.display = 'block';
hasResults = true;
} else {
item.style.display = 'none';
}
});
if (searchTerm && !hasResults) {
console.log('No results found');
}
});
});
</script>
<script id="scroll-animations">
document.addEventListener('DOMContentLoaded', function() {
const faqItems = document.querySelectorAll('.faq-item');
const observer = new IntersectionObserver((entries) => {
entries.forEach(entry => {
if (entry.isIntersecting) {
entry.target.style.opacity = '1';
entry.target.style.transform = 'translateY(0)';
}
});
}, {
threshold: 0.1,
rootMargin: '0px 0px -50px 0px'
});
faqItems.forEach((item, index) => {
item.style.opacity = '0';
item.style.transform = 'translateY(20px)';
item.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
observer.observe(item);
});
});
</script>
<script id="mobile-menu">
document.addEventListener('DOMContentLoaded', function() {
const menuButton = document.querySelector('button[class*="md:hidden"]');
if (menuButton) {
menuButton.addEventListener('click', function() {
console.log('Mobile menu clicked');
});
}
});
</script>
</body>
</html>