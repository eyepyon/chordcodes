<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>技術仕様 - Chord.codes AI 音楽生成技術</title>
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
<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.0/echarts.min.js"></script>
<style>
:where([class^="ri-"])::before {
content: "\f3c2";
}
.gradient-bg {
background: linear-gradient(135deg, #2E1B69 0%, #1B69B6 50%, #1BB6B6 100%);
}
.text-gradient {
background: linear-gradient(135deg, #1BB6B6, #1B69B6);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
}
.accordion-content {
max-height: 0;
overflow: hidden;
transition: max-height 0.3s ease-out;
}
.accordion-content.active {
max-height: 2000px;
}
.spec-card {
background: linear-gradient(145deg, #1f2937 0%, #111827 100%);
border: 1px solid rgba(27, 182, 182, 0.1);
}
.spec-card:hover {
border-color: rgba(27, 182, 182, 0.3);
transform: translateY(-2px);
}
</style>
</head>
<body class="bg-gray-900 text-white">
<header class="gradient-bg">
<nav class="flex items-center justify-between px-8 py-6">
<div class="flex items-center space-x-3">
<div class="w-10 h-10 flex items-center justify-center bg-white rounded-lg">
<i class="ri-music-2-fill text-primary text-xl"></i>
</div>
<span class="font-['Pacifico'] text-2xl text-white">Chord.codes</span>
</div>
<div class="hidden md:flex items-center space-x-8">
<a href="/" data-readdy="true" class="text-white hover:text-secondary transition-colors">機能</a>
<a href="#" class="text-secondary font-semibold">技術</a>
<a href="/" data-readdy="true" class="text-white hover:text-secondary transition-colors">活用例</a>
<a href="/" data-readdy="true" class="text-white hover:text-secondary transition-colors">体験</a>
</div>
<div class="flex items-center space-x-4">
<a href="/" data-readdy="true" class="flex items-center space-x-2 text-white hover:text-secondary transition-colors">
<div class="w-6 h-6 flex items-center justify-center">
<i class="ri-arrow-left-line text-lg"></i>
</div>
<span class="hidden md:inline">戻る</span>
</a>
<button class="md:hidden w-8 h-8 flex items-center justify-center text-white">
<i class="ri-menu-line text-xl"></i>
</button>
</div>
</nav>
</header>

<main class="min-h-screen">
<section class="py-16 bg-gray-900">
<div class="max-w-6xl mx-auto px-8">
<div class="text-center mb-16">
<h1 class="text-5xl font-bold mb-6">
<span class="text-gradient">技術仕様</span>
</h1>
<p class="text-xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
Chord.codes の革新的な AI 音楽生成技術の詳細仕様と実装について、包括的な技術ドキュメントをご覧ください。独自開発のアルゴリズムから軽量 MML 拡張技術まで、すべての技術的詳細を公開しています。
</p>
</div>

<div class="grid lg:grid-cols-4 gap-8">
<div class="lg:col-span-1">
<div class="sticky top-8 bg-gray-800 rounded-2xl p-6">
<h3 class="text-lg font-semibold mb-4 text-secondary">目次</h3>
<nav class="space-y-2">
<a href="#ai-model" class="block text-gray-300 hover:text-white transition-colors py-2 border-l-2 border-transparent hover:border-secondary pl-4">AI モデル詳細</a>
<a href="#mml-tech" class="block text-gray-300 hover:text-white transition-colors py-2 border-l-2 border-transparent hover:border-secondary pl-4">MML 拡張技術</a>
<a href="#hardware" class="block text-gray-300 hover:text-white transition-colors py-2 border-l-2 border-transparent hover:border-secondary pl-4">ハードウェア要件</a>
<a href="#generation-process" class="block text-gray-300 hover:text-white transition-colors py-2 border-l-2 border-transparent hover:border-secondary pl-4">音楽生成プロセス</a>
<a href="#performance" class="block text-gray-300 hover:text-white transition-colors py-2 border-l-2 border-transparent hover:border-secondary pl-4">パフォーマンス指標</a>
<a href="#compatibility" class="block text-gray-300 hover:text-white transition-colors py-2 border-l-2 border-transparent hover:border-secondary pl-4">対応デバイス</a>
</nav>
</div>
</div>

<div class="lg:col-span-3 space-y-12">
<section id="ai-model" class="spec-card rounded-2xl p-8 transition-all duration-300">
<div class="flex items-center justify-between mb-6 cursor-pointer accordion-trigger" data-target="ai-model-content">
<h2 class="text-3xl font-bold flex items-center space-x-3">
<div class="w-10 h-10 flex items-center justify-center bg-gradient-to-br from-primary to-secondary rounded-lg">
<i class="ri-brain-line text-white text-lg"></i>
</div>
<span>AI モデル詳細</span>
</h2>
<div class="w-8 h-8 flex items-center justify-center">
<i class="ri-arrow-down-s-line text-2xl accordion-icon transition-transform"></i>
</div>
</div>
<div id="ai-model-content" class="accordion-content active">
<div class="grid md:grid-cols-2 gap-8 mb-8">
<div>
<h3 class="text-xl font-semibold mb-4 text-secondary">使用アルゴリズム</h3>
<div class="space-y-4">
<div class="bg-gray-800 p-4 rounded-lg">
<h4 class="font-semibold mb-2">Transformer ベースアーキテクチャ</h4>
<p class="text-gray-300 text-sm">独自改良した Multi-Head Attention メカニズムにより、音楽的文脈を深く理解</p>
</div>
<div class="bg-gray-800 p-4 rounded-lg">
<h4 class="font-semibold mb-2">RAG (Retrieval-Augmented Generation)</h4>
<p class="text-gray-300 text-sm">50 万曲以上のクラシック楽曲データベースから最適な要素を動的に抽出</p>
</div>
<div class="bg-gray-800 p-4 rounded-lg">
<h4 class="font-semibold mb-2">リアルタイム推論エンジン</h4>
<p class="text-gray-300 text-sm">ONNX Runtime 最適化により、モバイルデバイスでも高速処理を実現</p>
</div>
</div>
</div>
<div>
<h3 class="text-xl font-semibold mb-4 text-secondary">学習データ概要</h3>
<div class="space-y-3">
<div class="flex justify-between items-center py-2 border-b border-gray-700">
<span class="text-gray-300">クラシック楽曲</span>
<span class="font-semibold">520,000 曲</span>
</div>
<div class="flex justify-between items-center py-2 border-b border-gray-700">
<span class="text-gray-300">MIDI データ</span>
<span class="font-semibold">1,200,000 ファイル</span>
</div>
<div class="flex justify-between items-center py-2 border-b border-gray-700">
<span class="text-gray-300">作曲家スタイル</span>
<span class="font-semibold">150 種類</span>
</div>
<div class="flex justify-between items-center py-2 border-b border-gray-700">
<span class="text-gray-300">学習時間</span>
<span class="font-semibold">2,400 時間</span>
</div>
<div class="flex justify-between items-center py-2">
<span class="text-gray-300">モデルサイズ</span>
<span class="font-semibold">45.2 MB</span>
</div>
</div>
</div>
</div>
<div class="bg-gray-800 p-6 rounded-xl">
<h3 class="text-lg font-semibold mb-4">モデルアーキテクチャ図</h3>
<div id="model-architecture" class="h-80 w-full"></div>
</div>
</div>
</section>

<section id="mml-tech" class="spec-card rounded-2xl p-8 transition-all duration-300">
<div class="flex items-center justify-between mb-6 cursor-pointer accordion-trigger" data-target="mml-content">
<h2 class="text-3xl font-bold flex items-center space-x-3">
<div class="w-10 h-10 flex items-center justify-center bg-gradient-to-br from-primary to-secondary rounded-lg">
<i class="ri-code-s-slash-line text-white text-lg"></i>
</div>
<span>MML 拡張技術</span>
</h2>
<div class="w-8 h-8 flex items-center justify-center">
<i class="ri-arrow-down-s-line text-2xl accordion-icon transition-transform"></i>
</div>
</div>
<div id="mml-content" class="accordion-content">
<div class="grid md:grid-cols-2 gap-8 mb-8">
<div>
<h3 class="text-xl font-semibold mb-4 text-secondary">独自拡張仕様</h3>
<div class="bg-gray-800 p-6 rounded-xl">
<h4 class="font-semibold mb-3">Extended MML (EMML) フォーマット</h4>
<div class="bg-gray-900 p-4 rounded-lg font-mono text-sm text-green-400 mb-4">
<div>// 基本構文例</div>
<div>T120 L4 V100</div>
<div>@STYLE:chopin @MOOD:melancholy</div>
<div>@DYNAMICS:pp-ff @PEDAL:auto</div>
<div>C D E F | G A B >C</div>
</div>
<ul class="space-y-2 text-gray-300 text-sm">
<li>• 動的テンポ変化対応</li>
<li>• 表情記号の詳細制御</li>
<li>• ペダリング自動最適化</li>
<li>• リアルタイム音色調整</li>
</ul>
</div>
</div>
<div>
<h3 class="text-xl font-semibold mb-4 text-secondary">データ圧縮効率</h3>
<div id="compression-chart" class="h-64 w-full mb-4"></div>
<div class="text-center text-sm text-gray-400">
従来 MIDI との比較（ファイルサイズ）
</div>
</div>
</div>
<div class="grid md:grid-cols-3 gap-6">
<div class="bg-gray-800 p-6 rounded-xl text-center">
<div class="text-3xl font-bold text-secondary mb-2">85%</div>
<div class="text-gray-300">ファイルサイズ削減</div>
</div>
<div class="bg-gray-800 p-6 rounded-xl text-center">
<div class="text-3xl font-bold text-secondary mb-2">3.2x</div>
<div class="text-gray-300">処理速度向上</div>
</div>
<div class="bg-gray-800 p-6 rounded-xl text-center">
<div class="text-3xl font-bold text-secondary mb-2">128</div>
<div class="text-gray-300">同時発音数</div>
</div>
</div>
</div>
</section>

<section id="hardware" class="spec-card rounded-2xl p-8 transition-all duration-300">
<div class="flex items-center justify-between mb-6 cursor-pointer accordion-trigger" data-target="hardware-content">
<h2 class="text-3xl font-bold flex items-center space-x-3">
<div class="w-10 h-10 flex items-center justify-center bg-gradient-to-br from-primary to-secondary rounded-lg">
<i class="ri-usb-line text-white text-lg"></i>
</div>
<span>ハードウェア要件</span>
</h2>
<div class="w-8 h-8 flex items-center justify-center">
<i class="ri-arrow-down-s-line text-2xl accordion-icon transition-transform"></i>
</div>
</div>
<div id="hardware-content" class="accordion-content">
<div class="grid md:grid-cols-2 gap-8 mb-8">
<div>
<h3 class="text-xl font-semibold mb-4 text-secondary">USB Type-C 接続仕様</h3>
<div class="space-y-4">
<div class="bg-gray-800 p-4 rounded-lg">
<div class="flex justify-between items-center mb-2">
<span class="font-semibold">USB 規格</span>
<span class="text-secondary">USB 3.2 Gen 1</span>
</div>
<div class="flex justify-between items-center mb-2">
<span class="font-semibold">データ転送速度</span>
<span class="text-secondary">5 Gbps</span>
</div>
<div class="flex justify-between items-center mb-2">
<span class="font-semibold">電力供給</span>
<span class="text-secondary">USB PD 3.0</span>
</div>
<div class="flex justify-between items-center">
<span class="font-semibold">MIDI over USB</span>
<span class="text-secondary">Class Compliant</span>
</div>
</div>
<div class="bg-gray-800 p-4 rounded-lg">
<h4 class="font-semibold mb-3">対応ケーブル仕様</h4>
<ul class="space-y-1 text-gray-300 text-sm">
<li>• USB Type-C to Type-C (推奨)</li>
<li>• USB Type-C to Type-B (変換アダプタ)</li>
<li>• 最大ケーブル長：3m</li>
<li>• シールド対応必須</li>
</ul>
</div>
</div>
<div>
<h3 class="text-xl font-semibold mb-4 text-secondary">システム要件</h3>
<div class="space-y-6">
<div>
<h4 class="font-semibold mb-3 flex items-center space-x-2">
<div class="w-6 h-6 flex items-center justify-center">
<i class="ri-smartphone-line text-secondary"></i>
</div>
<span>モバイルデバイス</span>
</h4>
<div class="bg-gray-800 p-4 rounded-lg space-y-2 text-sm">
<div class="flex justify-between">
<span>iOS</span>
<span>15.0 以降</span>
</div>
<div class="flex justify-between">
<span>Android</span>
<span>API Level 28 以降</span>
</div>
<div class="flex justify-between">
<span>RAM</span>
<span>4GB 以上推奨</span>
</div>
<div class="flex justify-between">
<span>ストレージ</span>
<span>200MB 以上</span>
</div>
</div>
</div>
<div>
<h4 class="font-semibold mb-3 flex items-center space-x-2">
<div class="w-6 h-6 flex items-center justify-center">
<i class="ri-piano-line text-secondary"></i>
</div>
<span>電子ピアノ</span>
</h4>
<div class="bg-gray-800 p-4 rounded-lg space-y-2 text-sm">
<div class="flex justify-between">
<span>MIDI 対応</span>
<span>必須</span>
</div>
<div class="flex justify-between">
<span>USB 接続</span>
<span>Type-A または Type-C</span>
</div>
<div class="flex justify-between">
<span>鍵盤数</span>
<span>88 鍵推奨</span>
</div>
<div class="flex justify-between">
<span>ベロシティ</span>
<span>128 段階対応</span>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</section>

<section id="generation-process" class="spec-card rounded-2xl p-8 transition-all duration-300">
<div class="flex items-center justify-between mb-6 cursor-pointer accordion-trigger" data-target="process-content">
<h2 class="text-3xl font-bold flex items-center space-x-3">
<div class="w-10 h-10 flex items-center justify-center bg-gradient-to-br from-primary to-secondary rounded-lg">
<i class="ri-flow-chart text-white text-lg"></i>
</div>
<span>音楽生成プロセス</span>
</h2>
<div class="w-8 h-8 flex items-center justify-center">
<i class="ri-arrow-down-s-line text-2xl accordion-icon transition-transform"></i>
</div>
</div>
<div id="process-content" class="accordion-content">
<div class="mb-8">
<h3 class="text-xl font-semibold mb-6 text-secondary">技術的フロー</h3>
<div class="grid md:grid-cols-4 gap-4">
<div class="bg-gray-800 p-6 rounded-xl text-center">
<div class="w-12 h-12 flex items-center justify-center bg-primary rounded-full mx-auto mb-4">
<span class="text-white font-bold">1</span>
</div>
<h4 class="font-semibold mb-2">スタイル解析</h4>
<p class="text-gray-400 text-sm">選択された作曲家スタイルの特徴を RAG ライブラリから抽出</p>
</div>
<div class="bg-gray-800 p-6 rounded-xl text-center">
<div class="w-12 h-12 flex items-center justify-center bg-primary rounded-full mx-auto mb-4">
<span class="text-white font-bold">2</span>
</div>
<h4 class="font-semibold mb-2">楽曲生成</h4>
<p class="text-gray-400 text-sm">Transformer モデルによる確率的楽曲構造の生成</p>
</div>
<div class="bg-gray-800 p-6 rounded-xl text-center">
<div class="w-12 h-12 flex items-center justify-center bg-primary rounded-full mx-auto mb-4">
<span class="text-white font-bold">3</span>
</div>
<h4 class="font-semibold mb-2">EMML 変換</h4>
<p class="text-gray-400 text-sm">生成された楽曲データを Extended MML 形式に最適化</p>
</div>
<div class="bg-gray-800 p-6 rounded-xl text-center">
<div class="w-12 h-12 flex items-center justify-center bg-primary rounded-full mx-auto mb-4">
<span class="text-white font-bold">4</span>
</div>
<h4 class="font-semibold mb-2">MIDI 出力</h4>
<p class="text-gray-400 text-sm">リアルタイム MIDI 信号として電子ピアノに送信</p>
</div>
</div>
</div>
<div class="bg-gray-800 p-6 rounded-xl">
<h3 class="text-lg font-semibold mb-4">処理時間分析</h3>
<div id="processing-time-chart" class="h-64 w-full"></div>
</div>
</div>
</section>

<section id="performance" class="spec-card rounded-2xl p-8 transition-all duration-300">
<div class="flex items-center justify-between mb-6 cursor-pointer accordion-trigger" data-target="performance-content">
<h2 class="text-3xl font-bold flex items-center space-x-3">
<div class="w-10 h-10 flex items-center justify-center bg-gradient-to-br from-primary to-secondary rounded-lg">
<i class="ri-speed-line text-white text-lg"></i>
</div>
<span>パフォーマンス指標</span>
</h2>
<div class="w-8 h-8 flex items-center justify-center">
<i class="ri-arrow-down-s-line text-2xl accordion-icon transition-transform"></i>
</div>
</div>
<div id="performance-content" class="accordion-content">
<div class="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
<div class="bg-gray-800 p-6 rounded-xl text-center">
<div class="text-3xl font-bold text-secondary mb-2">< 5ms</div>
<div class="text-gray-300 mb-2">レイテンシー</div>
<div class="text-xs text-gray-500">USB 接続時</div>
</div>
<div class="bg-gray-800 p-6 rounded-xl text-center">
<div class="text-3xl font-bold text-secondary mb-2">99.9%</div>
<div class="text-gray-300 mb-2">稼働率</div>
<div class="text-xs text-gray-500">24 時間連続</div>
</div>
<div class="bg-gray-800 p-6 rounded-xl text-center">
<div class="text-3xl font-bold text-secondary mb-2">15%</div>
<div class="text-gray-300 mb-2">CPU 使用率</div>
<div class="text-xs text-gray-500">平均値</div>
</div>
<div class="bg-gray-800 p-6 rounded-xl text-center">
<div class="text-3xl font-bold text-secondary mb-2">120MB</div>
<div class="text-gray-300 mb-2">メモリ使用量</div>
<div class="text-xs text-gray-500">最大値</div>
</div>
</div>
<div class="bg-gray-800 p-6 rounded-xl">
<h3 class="text-lg font-semibold mb-4">リアルタイム性能監視</h3>
<div id="performance-chart" class="h-80 w-full"></div>
</div>
</div>
</section>

<section id="compatibility" class="spec-card rounded-2xl p-8 transition-all duration-300">
<div class="flex items-center justify-between mb-6 cursor-pointer accordion-trigger" data-target="compatibility-content">
<h2 class="text-3xl font-bold flex items-center space-x-3">
<div class="w-10 h-10 flex items-center justify-center bg-gradient-to-br from-primary to-secondary rounded-lg">
<i class="ri-device-line text-white text-lg"></i>
</div>
<span>対応デバイス一覧</span>
</h2>
<div class="w-8 h-8 flex items-center justify-center">
<i class="ri-arrow-down-s-line text-2xl accordion-icon transition-transform"></i>
</div>
</div>
<div id="compatibility-content" class="accordion-content">
<div class="grid md:grid-cols-2 gap-8">
<div>
<h3 class="text-xl font-semibold mb-4 text-secondary">推奨電子ピアノ</h3>
<div class="space-y-3">
<div class="bg-gray-800 p-4 rounded-lg flex justify-between items-center">
<div>
<div class="font-semibold">Yamaha P-125</div>
<div class="text-sm text-gray-400">USB Type-B 接続</div>
</div>
<div class="w-6 h-6 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
<div class="bg-gray-800 p-4 rounded-lg flex justify-between items-center">
<div>
<div class="font-semibold">Roland FP-30X</div>
<div class="text-sm text-gray-400">USB Type-B 接続</div>
</div>
<div class="w-6 h-6 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
<div class="bg-gray-800 p-4 rounded-lg flex justify-between items-center">
<div>
<div class="font-semibold">Casio PX-S3100</div>
<div class="text-sm text-gray-400">USB Type-C 接続</div>
</div>
<div class="w-6 h-6 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
<div class="bg-gray-800 p-4 rounded-lg flex justify-between items-center">
<div>
<div class="font-semibold">Korg B2SP</div>
<div class="text-sm text-gray-400">USB Type-B 接続</div>
</div>
<div class="w-6 h-6 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
<div class="bg-gray-800 p-4 rounded-lg flex justify-between items-center">
<div>
<div class="font-semibold">Kawai ES120</div>
<div class="text-sm text-gray-400">USB Type-B 接続</div>
</div>
<div class="w-6 h-6 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
</div>
</div>
<div>
<h3 class="text-xl font-semibold mb-4 text-secondary">対応スマートフォン</h3>
<div class="space-y-4">
<div>
<h4 class="font-semibold mb-3 flex items-center space-x-2">
<div class="w-6 h-6 flex items-center justify-center">
<i class="ri-apple-fill text-gray-400"></i>
</div>
<span>iOS デバイス</span>
</h4>
<div class="bg-gray-800 p-4 rounded-lg space-y-2 text-sm">
<div class="flex justify-between">
<span>iPhone 12 以降</span>
<div class="w-4 h-4 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
<div class="flex justify-between">
<span>iPad Pro (2021 以降)</span>
<div class="w-4 h-4 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
<div class="flex justify-between">
<span>iPad Air (第 4 世代以降)</span>
<div class="w-4 h-4 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
</div>
</div>
<div>
<h4 class="font-semibold mb-3 flex items-center space-x-2">
<div class="w-6 h-6 flex items-center justify-center">
<i class="ri-android-fill text-gray-400"></i>
</div>
<span>Android デバイス</span>
</h4>
<div class="bg-gray-800 p-4 rounded-lg space-y-2 text-sm">
<div class="flex justify-between">
<span>Samsung Galaxy S21 以降</span>
<div class="w-4 h-4 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
<div class="flex justify-between">
<span>Google Pixel 6 以降</span>
<div class="w-4 h-4 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
<div class="flex justify-between">
<span>OnePlus 9 以降</span>
<div class="w-4 h-4 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
<div class="flex justify-between">
<span>Xiaomi Mi 11 以降</span>
<div class="w-4 h-4 flex items-center justify-center text-green-400">
<i class="ri-check-line"></i>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</section>
</div>
</div>
</div>
</section>
</main>

<footer class="bg-gray-900 border-t border-gray-800 py-12">
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
<li><a href="/" data-readdy="true" class="hover:text-white transition-colors">機能紹介</a></li>
<li><a href="#" class="text-secondary">技術仕様</a></li>
<li><a href="#" class="hover:text-white transition-colors">価格</a></li>
<li><a href="faq.php" class="hover:text-white transition-colors">FAQ</a></li>
</ul>
</div>
<div>
<h3 class="font-semibold text-white mb-4">サポート</h3>
<ul class="space-y-2 text-gray-400">
<li><a href="support.php" class="hover:text-white transition-colors">お問い合わせ</a></li>
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

<script id="accordion-functionality">
document.addEventListener('DOMContentLoaded', function() {
const triggers = document.querySelectorAll('.accordion-trigger');
triggers.forEach(trigger => {
trigger.addEventListener('click', function() {
const targetId = this.getAttribute('data-target');
const content = document.getElementById(targetId);
const icon = this.querySelector('.accordion-icon');

if (content.classList.contains('active')) {
content.classList.remove('active');
icon.style.transform = 'rotate(0deg)';
} else {
content.classList.add('active');
icon.style.transform = 'rotate(180deg)';
}
});
});
});
</script>

<script id="smooth-scroll-navigation">
document.addEventListener('DOMContentLoaded', function() {
const navLinks = document.querySelectorAll('nav a[href^="#"]');
navLinks.forEach(link => {
link.addEventListener('click', function(e) {
e.preventDefault();
const targetId = this.getAttribute('href');
const targetElement = document.querySelector(targetId);
if (targetElement) {
targetElement.scrollIntoView({
behavior: 'smooth',
block: 'start'
});
}
});
});
});
</script>

<script id="charts-initialization">
document.addEventListener('DOMContentLoaded', function() {
const modelArchChart = echarts.init(document.getElementById('model-architecture'));
const modelArchOption = {
animation: false,
tooltip: {
trigger: 'item',
backgroundColor: 'rgba(255, 255, 255, 0.9)',
textStyle: { color: '#1f2937' }
},
series: [{
type: 'graph',
layout: 'force',
data: [
{ name: 'Input Layer', x: 100, y: 300, symbolSize: 60, itemStyle: { color: 'rgba(87, 181, 231, 1)' }},
{ name: 'Embedding', x: 250, y: 300, symbolSize: 50, itemStyle: { color: 'rgba(141, 211, 199, 1)' }},
{ name: 'Multi-Head\nAttention', x: 400, y: 200, symbolSize: 70, itemStyle: { color: 'rgba(251, 191, 114, 1)' }},
{ name: 'Feed Forward', x: 400, y: 400, symbolSize: 60, itemStyle: { color: 'rgba(252, 141, 98, 1)' }},
{ name: 'RAG Module', x: 550, y: 150, symbolSize: 65, itemStyle: { color: 'rgba(87, 181, 231, 1)' }},
{ name: 'Output Layer', x: 700, y: 300, symbolSize: 60, itemStyle: { color: 'rgba(141, 211, 199, 1)' }}
],
links: [
{ source: 'Input Layer', target: 'Embedding' },
{ source: 'Embedding', target: 'Multi-Head\nAttention' },
{ source: 'Embedding', target: 'Feed Forward' },
{ source: 'Multi-Head\nAttention', target: 'RAG Module' },
{ source: 'Feed Forward', target: 'Output Layer' },
{ source: 'RAG Module', target: 'Output Layer' }
],
force: { repulsion: 200, edgeLength: 150 },
label: { show: true, color: '#ffffff', fontSize: 12 }
}],
backgroundColor: 'transparent'
};
modelArchChart.setOption(modelArchOption);

const compressionChart = echarts.init(document.getElementById('compression-chart'));
const compressionOption = {
animation: false,
tooltip: {
trigger: 'axis',
backgroundColor: 'rgba(255, 255, 255, 0.9)',
textStyle: { color: '#1f2937' }
},
xAxis: {
type: 'category',
data: ['MIDI', 'EMML', '圧縮率'],
axisLabel: { color: '#9ca3af' }
},
yAxis: {
type: 'value',
axisLabel: { color: '#9ca3af' }
},
series: [{
type: 'bar',
data: [100, 15, 85],
itemStyle: {
color: function(params) {
const colors = ['rgba(87, 181, 231, 1)', 'rgba(141, 211, 199, 1)', 'rgba(251, 191, 114, 1)'];
return colors[params.dataIndex];
},
borderRadius: [4, 4, 0, 0]
}
}],
grid: { top: 20, right: 20, bottom: 40, left: 40 },
backgroundColor: 'transparent'
};
compressionChart.setOption(compressionOption);

const processingTimeChart = echarts.init(document.getElementById('processing-time-chart'));
const processingTimeOption = {
animation: false,
tooltip: {
trigger: 'axis',
backgroundColor: 'rgba(255, 255, 255, 0.9)',
textStyle: { color: '#1f2937' }
},
xAxis: {
type: 'category',
data: ['スタイル解析', '楽曲生成', 'EMML変換', 'MIDI出力'],
axisLabel: { color: '#9ca3af', fontSize: 10 }
},
yAxis: {
type: 'value',
name: 'ms',
axisLabel: { color: '#9ca3af' }
},
series: [{
type: 'line',
data: [12, 45, 8, 2],
smooth: true,
lineStyle: { color: 'rgba(87, 181, 231, 1)', width: 3 },
itemStyle: { color: 'rgba(87, 181, 231, 1)' },
areaStyle: { color: 'rgba(87, 181, 231, 0.1)' }
}],
grid: { top: 40, right: 20, bottom: 60, left: 50 },
backgroundColor: 'transparent'
};
processingTimeChart.setOption(processingTimeOption);

const performanceChart = echarts.init(document.getElementById('performance-chart'));
const performanceOption = {
animation: false,
tooltip: {
trigger: 'axis',
backgroundColor: 'rgba(255, 255, 255, 0.9)',
textStyle: { color: '#1f2937' }
},
legend: {
data: ['CPU使用率', 'メモリ使用量', 'レイテンシー'],
textStyle: { color: '#9ca3af' }
},
xAxis: {
type: 'category',
data: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
axisLabel: { color: '#9ca3af' }
},
yAxis: [{
type: 'value',
name: '%',
axisLabel: { color: '#9ca3af' }
}, {
type: 'value',
name: 'ms',
axisLabel: { color: '#9ca3af' }
}],
series: [{
name: 'CPU使用率',
type: 'line',
data: [12, 15, 18, 22, 19, 16, 13],
smooth: true,
lineStyle: { color: 'rgba(87, 181, 231, 1)' },
itemStyle: { color: 'rgba(87, 181, 231, 1)' }
}, {
name: 'メモリ使用量',
type: 'line',
data: [85, 92, 88, 95, 90, 87, 84],
smooth: true,
lineStyle: { color: 'rgba(141, 211, 199, 1)' },
itemStyle: { color: 'rgba(141, 211, 199, 1)' }
}, {
name: 'レイテンシー',
type: 'line',
yAxisIndex: 1,
data: [3.2, 4.1, 3.8, 4.5, 3.9, 3.5, 3.1],
smooth: true,
lineStyle: { color: 'rgba(251, 191, 114, 1)' },
itemStyle: { color: 'rgba(251, 191, 114, 1)' }
}],
grid: { top: 60, right: 80, bottom: 40, left: 60 },
backgroundColor: 'transparent'
};
performanceChart.setOption(performanceOption);

window.addEventListener('resize', function() {
modelArchChart.resize();
compressionChart.resize();
processingTimeChart.resize();
performanceChart.resize();
});
});
</script>

</body>
</html>