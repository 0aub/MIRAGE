import { useState, useRef, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, Bot, User, ChevronDown, Loader2 } from "lucide-react";
import { Network } from "vis-network";
import { chatApi } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  timestamp: Date;
  retrievedNodes?: number;
  responseTime?: number;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "مرحباً! أنا مساعد MIRAGE. كيف يمكنني مساعدتك اليوم؟\n\nHello! I'm MIRAGE assistant. How can I help you today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isRTL, setIsRTL] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [contextGraph, setContextGraph] = useState<{ nodes: any[]; edges: any[] }>({
    nodes: [],
    edges: [],
  });
  const [retrievedNodesCount, setRetrievedNodesCount] = useState(0);
  const networkRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Update graph visualization when context changes
  useEffect(() => {
    if (!networkRef.current || contextGraph.nodes.length === 0) return;

    const data = {
      nodes: contextGraph.nodes,
      edges: contextGraph.edges,
    };

    const options: any = {
      nodes: {
        shape: "dot",
        size: 20,
        font: {
          size: 11,
          face: "Noto Naskh Arabic, Inter",
          color: "#fff",
        },
        borderWidth: 2,
      },
      edges: {
        width: 1,
        smooth: true,
      },
      groups: {
        document: {
          color: {
            background: "hsl(210, 100%, 55%)",
            border: "hsl(210, 100%, 40%)",
          },
        },
        entity: {
          color: {
            background: "hsl(140, 65%, 45%)",
            border: "hsl(140, 65%, 35%)",
          },
        },
        concept: {
          color: {
            background: "hsl(25, 95%, 55%)",
            border: "hsl(25, 95%, 40%)",
          },
        },
      },
      physics: {
        enabled: false,
      },
      interaction: {
        dragNodes: false,
        dragView: false,
        zoomView: false,
      },
    };

    const network = new Network(networkRef.current, data, options);
    network.fit();

    return () => {
      network.destroy();
    };
  }, [contextGraph]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // Call backend API for detailed query response
      const response = await chatApi.queryDetailed(input, conversationId);

      // Update conversation ID if provided
      if (!conversationId && response.workflow_metadata?.conversation_id) {
        setConversationId(response.workflow_metadata.conversation_id);
      }

      // Extract sources from citations
      const sources = response.citations?.map(
        (citation: any) => citation.source || citation.document_id || "Unknown"
      ) || [];

      // Create assistant message
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.response || "No response received",
        sources: sources.length > 0 ? sources : undefined,
        timestamp: new Date(),
        retrievedNodes: response.workflow_metadata?.retrieved_nodes_count || 0,
        responseTime: response.workflow_metadata?.response_time_ms,
      };

      setMessages((prev) => [...prev, aiMessage]);

      // Update context graph if available
      if (response.graph_visualization?.nodes && response.graph_visualization.nodes.length > 0) {
        const nodes = response.graph_visualization.nodes.map((node: any, idx: number) => ({
          id: idx + 1,
          label: node.label || node.id,
          group: node.type?.toLowerCase() || "entity",
        }));

        const nodeIdMap = new Map(response.graph_visualization.nodes.map((node: any, idx: number) => [node.id, idx + 1]));

        const edges = response.graph_visualization.edges?.map((edge: any) => ({
          from: nodeIdMap.get(edge.source) || 1,
          to: nodeIdMap.get(edge.target) || 1,
        })) || [];

        setContextGraph({ nodes, edges });
        setRetrievedNodesCount(nodes.length);
      }

    } catch (error: any) {
      console.error('Error sending message:', error);

      // Show error message in chat
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Error: ${error.message || "Failed to get response from server. Please try again."}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);

      toast({
        title: "Error",
        description: "Failed to send message to server",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="h-[calc(100vh-12rem)] flex flex-col lg:flex-row gap-6 animate-fade-in pb-20 md:pb-0">
      {/* Chat Section */}
      <Card className="flex-1 flex flex-col p-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold">Chat</h1>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsRTL(!isRTL)}
          >
            {isRTL ? "EN" : "AR"}
          </Button>
        </div>

        <ScrollArea className="flex-1 pr-4" ref={scrollRef}>
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                dir={isRTL ? "rtl" : "ltr"}
              >
                {message.role === "assistant" && (
                  <div className="w-8 h-8 rounded-full gradient-primary flex items-center justify-center flex-shrink-0">
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                )}

                <div className={`flex flex-col gap-2 max-w-[80%]`}>
                  <Card
                    className={`p-4 ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-card"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  </Card>

                  {message.sources && (
                    <details className="text-xs">
                      <summary className="cursor-pointer text-muted-foreground hover:text-foreground flex items-center gap-1">
                        <ChevronDown className="w-3 h-3" />
                        Show Sources ({message.sources.length})
                      </summary>
                      <div className="mt-2 space-y-1 pl-4">
                        {message.sources.map((source, idx) => (
                          <div
                            key={idx}
                            className="text-muted-foreground hover:text-foreground"
                          >
                            • {source}
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>

                {message.role === "user" && (
                  <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center flex-shrink-0">
                    <User className="w-5 h-5" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </ScrollArea>

        <div className="mt-4 flex gap-2" dir={isRTL ? "rtl" : "ltr"}>
          <Input
            placeholder={isRTL ? "اكتب رسالتك هنا..." : "Type your message..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            className="flex-1"
            disabled={isLoading}
          />
          <Button onClick={handleSend} size="icon" disabled={isLoading}>
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </div>
      </Card>

      {/* Active Subgraph */}
      <Card className="lg:w-96 p-6">
        <h2 className="text-lg font-semibold mb-4">Active Context</h2>
        {contextGraph.nodes.length > 0 ? (
          <>
            <div
              ref={networkRef}
              className="w-full h-64 rounded-lg bg-secondary/20"
              style={{ border: "1px solid hsl(var(--border))" }}
            />
            <div className="mt-4 space-y-2">
              <p className="text-sm font-medium">Retrieved Nodes: {retrievedNodesCount}</p>
              <p className="text-sm text-muted-foreground">
                Showing relevant subgraph for current query
              </p>
            </div>
          </>
        ) : (
          <div className="w-full h-64 rounded-lg bg-secondary/20 flex items-center justify-center" style={{ border: "1px solid hsl(var(--border))" }}>
            <div className="text-center p-4">
              <p className="text-sm text-muted-foreground">
                {isLoading
                  ? "Retrieving context..."
                  : "Send a message to see the relevant knowledge graph context"
                }
              </p>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
