(function() {var implementors = {
"domain_client_consensus_relay_chain":[["impl&lt;Block&gt; Verifier&lt;Block&gt; for <a class=\"struct\" href=\"domain_client_consensus_relay_chain/struct.Verifier.html\" title=\"struct domain_client_consensus_relay_chain::Verifier\">Verifier</a>&lt;Block&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Block: BlockT,</span>"]],
"sc_consensus_subspace":[["impl&lt;Block, Client, SelectChain, SN&gt; Verifier&lt;Block&gt; for <a class=\"struct\" href=\"sc_consensus_subspace/struct.SubspaceVerifier.html\" title=\"struct sc_consensus_subspace::SubspaceVerifier\">SubspaceVerifier</a>&lt;Block, Client, SelectChain, SN&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Block: BlockT,<br>&nbsp;&nbsp;&nbsp;&nbsp;Client: HeaderBackend&lt;Block&gt; + ProvideRuntimeApi&lt;Block&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> + AuxStore,<br>&nbsp;&nbsp;&nbsp;&nbsp;Client::Api: BlockBuilderApi&lt;Block&gt; + SubspaceApi&lt;Block, FarmerPublicKey&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;SelectChain: SelectChain&lt;Block&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;SN: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/ops/function/trait.Fn.html\" title=\"trait core::ops::function::Fn\">Fn</a>() -&gt; Slot + <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> + 'static,</span>"]],
"sc_network_test":[["impl&lt;B:&nbsp;BlockT&gt; Verifier&lt;B&gt; for <a class=\"struct\" href=\"sc_network_test/struct.PassThroughVerifier.html\" title=\"struct sc_network_test::PassThroughVerifier\">PassThroughVerifier</a>"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()