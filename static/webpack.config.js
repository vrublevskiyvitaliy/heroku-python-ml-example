function getStyleUse(bundleFilename) {
  return [
    {
      loader: 'file-loader',
      options: {
        name: bundleFilename,
      },
    },
    { loader: 'extract-loader' },
    { loader: 'css-loader' },
    {
      loader: 'sass-loader',
      options: {
        includePaths: ['./node_modules'],
        implementation: require('dart-sass'),
        fiber: require('fibers'),
  }
    },
  ];
}

module.exports = [
  {
    entry: './material/main.scss',
    output: {
      // This is necessary for webpack to compile, but we never reference this js file.
      filename: 'not_used_main.js',
    },
    module: {
      rules: [{
        test: /main.scss$/,
        use: getStyleUse('./material/build/bundle-main.css')
      }]
    },
  },
  {
    entry: "./material/main.js",
    output: {
      filename: "material/build/bundle-main.js"
    },
    module: {
      loaders: [{
        test: /main.js$/,
        loader: 'babel-loader',
        query: {presets: ['env']}
      }]
    },
  },
];
