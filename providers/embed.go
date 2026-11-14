// Package providers holds the built-in provider data files.
package providers

import _ "embed"

// Active is the default list of providers the gateway serves.
//
//go:embed providers.json
var Active []byte
