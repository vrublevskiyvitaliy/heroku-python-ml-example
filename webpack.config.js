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
    entry: './static/material/main.scss',
    output: {
      // This is necessary for webpack to compile, but we never reference this js file.
      filename: './static/material/main.js',
    },
    module: {
      rules: [{
        test: /main.scss$/,
        use: getStyleUse('bundle-main.css')
      }]
    },
  },
  {
    entry: "./static/material/main.js",
    output: {
      filename: "bundle-main.js"
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
